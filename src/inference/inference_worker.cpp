#include "inference/inference_worker.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>

#ifdef _WIN32
#include <windows.h>
#endif

#include "inference/backend_registry.h"
#include "preprocessing/preprocess.h"
#include "utils/logging.h"
#include "utils/timer.h"

namespace promatte {

namespace {

// Observers run caller code on the worker thread; an exception there must not
// take the inference loop down.
template<typename Fn> void callObserver(const char *name, Fn &&fn)
{
	try {
		fn();
	} catch (const std::exception &e) {
		PM_LOG_ERROR("%s observer threw: %s", name, e.what());
	} catch (...) {
		PM_LOG_ERROR("%s observer threw an unknown exception", name);
	}
}

} // namespace

const char *workerStateName(WorkerState s)
{
	switch (s) {
	case WorkerState::Stopped:
		return "Stopped";
	case WorkerState::Idle:
		return "Idle";
	case WorkerState::Initializing:
		return "Initializing";
	case WorkerState::Running:
		return "Running";
	case WorkerState::Error:
		return "Error";
	case WorkerState::NoModel:
		return "No model";
	}
	return "?";
}

InferenceWorker::InferenceWorker() = default;

InferenceWorker::~InferenceWorker()
{
	stop();
}

void InferenceWorker::start()
{
	if (running_.exchange(true))
		return;
	stopping_ = false;
	setState(WorkerState::Idle);
	thread_ = std::thread([this] { threadMain(); });
}

void InferenceWorker::stop()
{
	stopping_ = true; // a backend still initialising on the worker is not installed
	wakeMatteWaiters(); // a render thread in waitForMatte() returns Aborted
	if (!running_.exchange(false)) {
		if (thread_.joinable())
			thread_.join();
		return;
	}
	{
		// The worker may be replacing backend_ right now; cancel whichever is current.
		std::lock_guard<std::mutex> lock(backendMutex_);
		if (backend_)
			backend_->cancel(); // abort an in-flight Run() so join is fast
	}
	cv_.notify_all();
	if (thread_.joinable())
		thread_.join();
	replaceBackend(nullptr);
	setState(WorkerState::Stopped);
}

void InferenceWorker::setState(WorkerState s, const std::string &error)
{
	{
		std::lock_guard<std::mutex> lock(statusMutex_);
		status_.state = s;
		status_.error = error;
	}
	stateFast_ = s;
	if (s != WorkerState::Running)
		wakeMatteWaiters();
}

void InferenceWorker::wakeMatteWaiters()
{
	// Taking the mutex orders the state change before a waiter's predicate check,
	// so the notification cannot fall between its check and its wait.
	{
		std::lock_guard<std::mutex> lock(matteMutex_);
	}
	matteCv_.notify_all();
}

void InferenceWorker::markFrameDone(uint64_t seq, MatteWait outcome)
{
	if (seq == 0)
		return;
	{
		std::lock_guard<std::mutex> lock(matteMutex_);
		if (seq >= doneSeq_) {
			doneSeq_ = seq;
			doneOutcome_ = outcome;
		}
	}
	matteCv_.notify_all();
}

MatteWait InferenceWorker::waitForMatte(uint64_t seq, double timeoutMs)
{
	const auto deadline = std::chrono::steady_clock::now() +
			      std::chrono::microseconds(int64_t(std::max(timeoutMs, 0.0) * 1000.0));
	std::unique_lock<std::mutex> lock(matteMutex_);
	MatteWait result = MatteWait::Timeout;
	auto ready = [&] {
		if (publishedSeq_ >= seq) {
			result = MatteWait::Ready;
			return true;
		}
		if (doneSeq_ >= seq) {
			result = doneSeq_ == seq ? doneOutcome_ : MatteWait::NoMatte;
			return true;
		}
		if (stopping_.load() || !running_.load() || stateFast_.load() != WorkerState::Running) {
			result = MatteWait::Aborted;
			return true;
		}
		return false;
	};
	if (!matteCv_.wait_until(lock, deadline, ready))
		return MatteWait::Timeout;
	return result;
}

void InferenceWorker::setConfig(const WorkerConfig &config)
{
	{
		std::lock_guard<std::mutex> lock(ctrlMutex_);
		pendingConfig_ = config;
		configPending_ = true;
		releasePending_ = false;
	}
	cv_.notify_all();
}

void InferenceWorker::setTuning(const TemporalParams &temporal, const RefineParams &refine)
{
	std::lock_guard<std::mutex> lock(ctrlMutex_);
	pendingTemporal_ = temporal;
	pendingRefine_ = refine;
	tuningPending_ = true;
}

void InferenceWorker::setSourceInfo(uint32_t width, uint32_t height, double fps)
{
	std::lock_guard<std::mutex> lock(ctrlMutex_);
	pendingSrcW_ = width;
	pendingSrcH_ = height;
	pendingSrcFps_ = fps;
	sourcePending_ = true;
}

void InferenceWorker::setQualityMode(QualityMode mode, int manualTier)
{
	std::lock_guard<std::mutex> lock(ctrlMutex_);
	pendingQuality_ = mode;
	pendingManualTier_ = manualTier;
	qualityPending_ = true;
}

void InferenceWorker::setMotionSkip(bool enabled, float threshold, double maxStaleMs)
{
	skipEnabled_ = enabled;
	skipThreshold_ = std::max(threshold, 0.f);
	skipMaxStaleMs_ = std::max(maxStaleMs, 0.0);
}

void InferenceWorker::setEnabled(bool enabled)
{
	enabled_ = enabled;
	if (enabled)
		cv_.notify_all();
}

void InferenceWorker::releaseBackend()
{
	{
		std::lock_guard<std::mutex> lock(ctrlMutex_);
		releasePending_ = true;
	}
	cv_.notify_all();
}

void InferenceWorker::setCostObserver(CostObserver observer)
{
	std::lock_guard<std::mutex> lock(observerMutex_);
	costObserver_ = std::move(observer);
}

void InferenceWorker::setModelChangeObserver(ModelChangeObserver observer)
{
	std::lock_guard<std::mutex> lock(observerMutex_);
	modelChangeObserver_ = std::move(observer);
}

void InferenceWorker::setBackendFallbackObserver(BackendFallbackObserver observer)
{
	std::lock_guard<std::mutex> lock(observerMutex_);
	backendFallbackObserver_ = std::move(observer);
}

void InferenceWorker::resetTemporal()
{
	std::lock_guard<std::mutex> lock(ctrlMutex_);
	resetTemporalPending_ = true;
}

void InferenceWorker::desiredResolution(uint32_t &w, uint32_t &h) const
{
	w = aiW_.load();
	h = aiH_.load();
}

bool InferenceWorker::wantsFrame() const
{
	if (!enabled_.load() || aiW_.load() == 0)
		return false;
	if (Stopwatch::nowMs() < nextDueMs_.load())
		return false; // maxAiFps throttle: no readback for a frame that would wait
	std::lock_guard<std::mutex> lock(frameMutex_);
	return !frameAvailable_;
}

uint64_t InferenceWorker::submitFrame(FrameBuffer &frame)
{
	bool overwrote = false;
	uint64_t seq = 0, overwrittenSeq = 0;
	{
		std::lock_guard<std::mutex> lock(frameMutex_);
		overwrote = frameAvailable_;
		if (overwrote)
			overwrittenSeq = frameSlot_.seq;
		seq = nextSeq_++;
		frame.seq = seq;
		frame.timestampMs = Stopwatch::nowMs();
		std::swap(frame, frameSlot_); // caller gets the old buffer back (no allocation churn)
		frameAvailable_ = true;
	}
	if (overwrote)
		markFrameDone(overwrittenSeq, MatteWait::NoMatte);
	perf_.recordSubmit(overwrote);
	perf_.recordQueueDepth(1);
	// The worker checks for a frame under ctrlMutex_ before it sleeps on cv_.
	// Taking that mutex here means it either sees the frame or is already
	// waiting, so the notification wakes it at once instead of after the 50 ms
	// idle timeout (which the synchronous mode would pay on every frame).
	{
		std::lock_guard<std::mutex> lock(ctrlMutex_);
	}
	cv_.notify_one();
	return seq;
}

bool InferenceWorker::takeMatte(MatteBuffer &out)
{
	if (!matteAvailable_.load())
		return false;
	std::lock_guard<std::mutex> lock(matteMutex_);
	if (matteSlot_.seq <= out.seq || matteSlot_.empty())
		return false;
	std::swap(out, matteSlot_);
	matteAvailable_ = false;
	return true;
}

WorkerStatus InferenceWorker::status() const
{
	std::lock_guard<std::mutex> lock(statusMutex_);
	return status_;
}

WorkerState InferenceWorker::state() const
{
	std::lock_guard<std::mutex> lock(statusMutex_);
	return status_.state;
}

PerfStats InferenceWorker::stats() const
{
	return perf_.snapshot();
}

void InferenceWorker::updateAiResolution()
{
	const ModelDescriptor &m = config_.model;
	if (m.tiers.empty()) {
		aiW_ = aiH_ = 0;
		return;
	}
	tier_ = std::clamp(tier_, 0, int(m.tiers.size()) - 1);
	const ResolutionTier &t = m.tiers[size_t(tier_)];
	uint32_t w = 0, h = 0;
	chooseAiResolution(config_.sourceWidth, config_.sourceHeight, t.width, t.height, m.fixedInput, m.alignment, w, h);
	aiW_ = w;
	aiH_ = h;
	ratio_ = t.downsampleRatio;
	{
		std::lock_guard<std::mutex> lock(statusMutex_);
		status_.aiWidth = w;
		status_.aiHeight = h;
		status_.tier = tier_;
		status_.tierLabel = t.label;
		status_.downsampleRatio = t.downsampleRatio;
	}
	if (backend_)
		backend_->resetState();
	temporal_.reset();
	thumbRefValid_ = false; // the next frame always runs the model
	settleFrames_ = kSettleFrames;
	PM_LOG_INFO("AI resolution: %ux%u (tier %d '%s', ratio %.3f) for source %ux%u", w, h, tier_, t.label.c_str(),
		    t.downsampleRatio, config_.sourceWidth, config_.sourceHeight);
}

std::unique_ptr<SegmentationBackend> InferenceWorker::initializeBackend(WorkerConfig &cfg, std::string &error)
{
	if (cfg.model.id.empty() || cfg.modelPath.empty()) {
		error = "no model selected";
		return nullptr;
	}
	BackendKind kind = cfg.backend.kind;
	if (kind == BackendKind::Auto)
		kind = resolveAutoBackend();
	std::vector<BackendKind> attempts{kind};
	if (cfg.backend.allowFallback) {
		if (kind == BackendKind::TensorRT)
			attempts.push_back(BackendKind::CUDA);
		if (kind != BackendKind::CPU && kind != BackendKind::DirectML && backendAvailable(BackendKind::DirectML))
			attempts.push_back(BackendKind::DirectML);
		if (kind != BackendKind::CPU)
			attempts.push_back(BackendKind::CPU);
	}
	const bool haveAlt = !cfg.altModel.id.empty() && !cfg.altModelPath.empty();
	std::string lastError;
	for (BackendKind k : attempts) {
		if (stopping_) {
			error = "worker stopping";
			return nullptr;
		}
		if (!backendAvailable(k)) {
			lastError = std::string(backendKindName(k)) + " not available";
			continue;
		}
		// GPU: the configured model first, then the other precision (FP16 that
		// the provider rejects -> FP32). CPU: the alternative first, because the
		// owner only sets one when `model` is the GPU-oriented variant.
		std::vector<bool> useAlt{false};
		if (haveAlt)
			useAlt = k == BackendKind::CPU ? std::vector<bool>{true, false} : std::vector<bool>{false, true};
		for (bool alt : useAlt) {
			const ModelDescriptor &model = alt ? cfg.altModel : cfg.model;
			const std::string &path = alt ? cfg.altModelPath : cfg.modelPath;
			auto b = createBackend(k);
			std::string err;
			if (b->initialize(cfg.backend, model, path, err)) {
				if (stopping_) {
					error = "worker stopping";
					return nullptr;
				}
				if (k != kind)
					PM_LOG_WARN("backend %s failed (%s); fell back to %s", backendKindName(kind),
						    lastError.c_str(), backendKindName(k));
				if (alt) {
					PM_LOG_INFO("using model '%s' instead of '%s' on %s", cfg.altModel.id.c_str(),
						    cfg.model.id.c_str(), backendKindName(k));
					std::swap(cfg.model, cfg.altModel);
					std::swap(cfg.modelPath, cfg.altModelPath);
				}
				return b;
			}
			lastError = err;
			PM_LOG_WARN("backend %s init failed for model '%s': %s", backendKindName(k), model.id.c_str(),
				    err.c_str());
		}
	}
	error = lastError.empty() ? "no inference backend available" : lastError;
	return nullptr;
}

bool InferenceWorker::replaceBackend(std::unique_ptr<SegmentationBackend> next)
{
	std::unique_ptr<SegmentationBackend> old;
	{
		std::lock_guard<std::mutex> lock(backendMutex_);
		if (next && stopping_)
			return false; // `next` is released outside the lock on return
		old = std::move(backend_);
		backend_ = std::move(next);
	}
	// Session teardown can take a while; never do it while stop() waits on the lock.
	old.reset();
	return true;
}

void InferenceWorker::publishModel(const WorkerConfig &cfg, bool usingFallback)
{
	std::lock_guard<std::mutex> lock(statusMutex_);
	status_.modelId = cfg.model.id;
	status_.modelName = cfg.model.displayName;
	status_.recurrent = cfg.model.recurrent;
	status_.providesForeground = cfg.model.providesForeground;
	status_.usingFallbackModel = usingFallback;
	if (backend_)
		status_.backend = backend_->capabilities();
}

bool InferenceWorker::applyConfigLocked(WorkerConfig cfg, bool keepCurrentOnFailure)
{
	// ctrlMutex_ is NOT held here (called from the worker thread after copying).
	// The owner re-posting the precision pair the worker already swapped (FP16
	// rejected, FP32 running) is the same configuration, not a new model.
	if (backend_ && cfg.altModel.id == config_.model.id && cfg.model.id == config_.altModel.id &&
	    cfg.altModelPath == config_.modelPath && !cfg.altModel.id.empty()) {
		std::swap(cfg.model, cfg.altModel);
		std::swap(cfg.modelPath, cfg.altModelPath);
	}
	const bool needBackend = !backend_ || cfg.model.id != config_.model.id || cfg.modelPath != config_.modelPath ||
				 cfg.backend.kind != config_.backend.kind ||
				 cfg.backend.deviceIndex != config_.backend.deviceIndex ||
				 cfg.backend.cpuThreads != config_.backend.cpuThreads;
	// Re-posting the model already in use (the owner refreshing fallback/upgrade
	// candidates) keeps the fallback flag the worker set when it switched.
	bool usingFallback = false;
	if (backend_ && cfg.model.id == config_.model.id) {
		std::lock_guard<std::mutex> lock(statusMutex_);
		usingFallback = status_.usingFallbackModel;
	}
	// A new configuration supersedes any pending error recovery.
	consecutiveErrors_ = 0;
	reinitDelayMs_ = 0;
	reinitDueMs_ = -1;
	if (needBackend) {
		if (!keepCurrentOnFailure) {
			// Release the old session first so two models never occupy (V)RAM at once.
			replaceBackend(nullptr);
			publishModel(cfg, false);
			setState(WorkerState::Initializing);
		}
		std::string err;
		std::unique_ptr<SegmentationBackend> fresh = initializeBackend(cfg, err);
		if (!fresh) {
			if (keepCurrentOnFailure && backend_) {
				PM_LOG_WARN("could not switch to model '%s' (%s); keeping '%s'", cfg.model.id.c_str(),
					    err.c_str(), config_.model.id.c_str());
				return false;
			}
			replaceBackend(nullptr);
			config_ = cfg;
			publishModel(cfg, false);
			setState(cfg.modelPath.empty() ? WorkerState::NoModel : WorkerState::Error, err);
			PM_LOG_ERROR("inference initialisation failed: %s", err.c_str());
			aiW_ = aiH_ = 0;
			return false;
		}
		if (!replaceBackend(std::move(fresh)))
			return false; // stopping
	}
	config_ = cfg;
	temporal_.setParams(cfg.temporal);
	publishModel(cfg, usingFallback);
	const bool gpu = backend_ && backend_->capabilities().isGpu;
	perf_.configure(cfg.quality, int(cfg.model.tiers.size()), cfg.sourceFps, gpu);
	tier_ = cfg.manualTier >= 0 ? std::clamp(cfg.manualTier, 0, int(cfg.model.tiers.size()) - 1)
				    : perf_.currentTier();
	if (cfg.manualTier >= 0)
		perf_.forceTier(tier_);
	int dummy;
	perf_.pollTierChange(dummy);
	updateAiResolution();
	setState(WorkerState::Running);
	return true;
}

void InferenceWorker::threadMain()
{
#ifdef _WIN32
	// OBS' render and encoder threads must win any contention for the CPU: on
	// low-core machines a CPU-backend inference otherwise starves the compositor.
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
	PM_LOG_DEBUG("inference worker thread started");
	while (running_) {
		uint64_t inFlightSeq = 0, droppedSeq = 0;
		try {
			// ---- control messages ------------------------------------------------
			bool haveConfig = false, haveTuning = false, haveSource = false, haveQuality = false, doRelease = false,
			     doResetTemporal = false;
			WorkerConfig cfg;
			TemporalParams tp;
			RefineParams rp;
			uint32_t sw = 0, sh = 0;
			double sfps = 30;
			QualityMode qm = QualityMode::Auto;
			int manualTier = -1;
			{
				std::unique_lock<std::mutex> lock(ctrlMutex_);
				bool frameReady = false;
				{
					std::lock_guard<std::mutex> flock(frameMutex_);
					frameReady = frameAvailable_;
				}
				if (!configPending_ && !tuningPending_ && !sourcePending_ && !qualityPending_ && !releasePending_ &&
				    !resetTemporalPending_ && !(frameReady && enabled_)) {
					cv_.wait_for(lock, std::chrono::milliseconds(50));
				}
				if (configPending_) {
					cfg = pendingConfig_;
					configPending_ = false;
					haveConfig = true;
				}
				if (tuningPending_) {
					tp = pendingTemporal_;
					rp = pendingRefine_;
					tuningPending_ = false;
					haveTuning = true;
				}
				if (sourcePending_) {
					sw = pendingSrcW_;
					sh = pendingSrcH_;
					sfps = pendingSrcFps_;
					sourcePending_ = false;
					haveSource = true;
				}
				if (qualityPending_) {
					qm = pendingQuality_;
					manualTier = pendingManualTier_;
					qualityPending_ = false;
					haveQuality = true;
				}
				if (releasePending_) {
					releasePending_ = false;
					doRelease = true;
				}
				if (resetTemporalPending_) {
					resetTemporalPending_ = false;
					doResetTemporal = true;
				}
			}
			if (!running_)
				break;
			if (doRelease) {
				replaceBackend(nullptr);
				config_.model = ModelDescriptor{};
				config_.modelPath.clear();
				aiW_ = aiH_ = 0;
				reinitDueMs_ = -1;
				consecutiveErrors_ = 0;
				setState(WorkerState::Idle);
			}
			if (haveConfig) {
				if (haveSource) {
					cfg.sourceWidth = sw;
					cfg.sourceHeight = sh;
					cfg.sourceFps = sfps;
					haveSource = false;
				}
				if (haveTuning) {
					cfg.temporal = tp;
					cfg.refine = rp;
					haveTuning = false;
				}
				if (haveQuality) {
					cfg.quality = qm;
					cfg.manualTier = manualTier;
					haveQuality = false;
				}
				applyConfigLocked(cfg);
			}
			// Backend dropped after repeated inference errors: retry once the back-off expires.
			if (!backend_ && reinitDueMs_ >= 0 && Stopwatch::nowMs() >= reinitDueMs_)
				tryReinitialize();
			if (haveTuning) {
				config_.temporal = tp;
				config_.refine = rp;
				temporal_.setParams(tp);
			}
			if (haveSource && (sw != config_.sourceWidth || sh != config_.sourceHeight ||
					   std::abs(sfps - config_.sourceFps) > 0.5)) {
				config_.sourceWidth = sw;
				config_.sourceHeight = sh;
				config_.sourceFps = sfps;
				if (backend_) {
					const bool gpu = backend_->capabilities().isGpu;
					int keepTier = tier_;
					perf_.configure(config_.quality, int(config_.model.tiers.size()), sfps, gpu);
					if (config_.manualTier >= 0)
						perf_.forceTier(config_.manualTier);
					else
						perf_.forceTier(keepTier);
					int t;
					if (perf_.pollTierChange(t))
						tier_ = t;
					updateAiResolution();
				}
			}
			if (haveQuality && backend_) {
				config_.quality = qm;
				config_.manualTier = manualTier;
				perf_.setMode(qm);
				if (manualTier >= 0)
					perf_.forceTier(manualTier);
				int t;
				if (perf_.pollTierChange(t)) {
					tier_ = t;
					updateAiResolution();
				}
			}
			if (doResetTemporal) {
				temporal_.reset();
				thumbRefValid_ = false;
				if (backend_)
					backend_->resetState();
			}
			// Adaptive tier changes decided by the performance controller.
			{
				int t;
				if (backend_ && perf_.pollTierChange(t) && t != tier_) {
					tier_ = t;
					updateAiResolution();
				}
			}
			// Automatic model changes (Auto mode only).
			if (backend_ && config_.modelAutoSelected) {
				const bool fallback = perf_.pollModelFallback();
				const bool upgrade = !fallback && perf_.pollModelUpgrade();
				if (fallback || upgrade) {
					const ModelDescriptor &target = fallback ? config_.fallbackModel : config_.upgradeModel;
					const std::string &targetPath = fallback ? config_.fallbackModelPath : config_.upgradeModelPath;
					if (!target.id.empty() && target.id != config_.model.id && !targetPath.empty()) {
						// Remember how the outgoing model performed before switching.
						reportCost(true);
						PM_LOG_INFO("switching model '%s' -> '%s' (%s)", config_.model.id.c_str(),
							    target.id.c_str(), fallback ? "too slow" : "headroom available");
						WorkerConfig cfg = config_;
						cfg.model = target;
						cfg.modelPath = targetPath;
						cfg.altModel = ModelDescriptor{}; // belonged to the outgoing model
						cfg.altModelPath.clear();
						cfg.fallbackModel = ModelDescriptor{};
						cfg.fallbackModelPath.clear();
						cfg.upgradeModel = ModelDescriptor{};
						cfg.upgradeModelPath.clear();
						// On failure the current model and backend stay in service.
						if (applyConfigLocked(cfg, /*keepCurrentOnFailure=*/true)) {
							{
								std::lock_guard<std::mutex> lock(statusMutex_);
								status_.usingFallbackModel = fallback;
							}
							ModelChangeObserver obs;
							{
								std::lock_guard<std::mutex> lock(observerMutex_);
								obs = modelChangeObserver_;
							}
							if (obs)
								callObserver("model change", [&] { obs(target.id); });
						}
					}
				}
			}
			reportCost(false);
			// GPU contention that survives every cheaper option: hand over to the owner.
			if (backend_ && backend_->capabilities().isGpu && config_.backend.kind == BackendKind::Auto &&
			    perf_.pollBackendFallback()) {
				const auto caps = backend_->capabilities();
				BackendFallbackObserver obs;
				{
					std::lock_guard<std::mutex> lock(observerMutex_);
					obs = backendFallbackObserver_;
				}
				if (obs)
					callObserver("backend fallback", [&] { obs(caps.deviceName + "|" + caps.name); });
			}

			// ---- frame processing --------------------------------------------------
			bool got = false;
			{
				std::lock_guard<std::mutex> lock(frameMutex_);
				if (frameAvailable_) {
					if (enabled_ && backend_) {
						std::swap(work_, frameSlot_);
						got = true;
					} else {
						droppedSeq = frameSlot_.seq;
					}
					frameAvailable_ = false; // drop when disabled / no backend
				}
			}
			perf_.recordQueueDepth(0);
			if (got) {
				inFlightSeq = work_.seq;
				lastOutcome_ = MatteWait::NoMatte; // stays so if processOne throws
				// Throttling happens in wantsFrame(): a frame that was read back is processed.
				processOne(work_);
			}
		} catch (const std::exception &e) {
			handleWorkerException(e.what());
		} catch (...) {
			handleWorkerException("unknown exception");
		}
		// Every frame taken from the slot is reported, whatever happened to it,
		// so a render thread waiting for it in waitForMatte() returns promptly.
		markFrameDone(droppedSeq, MatteWait::NoMatte);
		markFrameDone(inFlightSeq, lastOutcome_);
	}
	replaceBackend(nullptr);
	PM_LOG_DEBUG("inference worker thread exiting");
}

void InferenceWorker::handleInferenceError(const std::string &error)
{
	const double now = Stopwatch::nowMs();
	// Count the failed attempt as processed so throttling and dt stay sane.
	lastProcessedMs_ = now;
	++consecutiveErrors_;
	if (shouldLogError(now)) {
		if (suppressedErrors_ > 0)
			PM_LOG_ERROR("inference error: %s (%llu similar errors suppressed)", error.c_str(), suppressedErrors_);
		else
			PM_LOG_ERROR("inference error: %s", error.c_str());
		suppressedErrors_ = 0;
	}
	if (consecutiveErrors_ < kErrorsBeforeReset)
		return;
	// The session is probably wedged (device removed, driver reset): drop it and
	// rebuild it after a growing pause instead of failing every frame.
	reinitDelayMs_ = reinitDelayMs_ <= 0 ? 1000.0 : std::min(reinitDelayMs_ * 2.0, kMaxReinitDelayMs);
	reinitDueMs_ = now + reinitDelayMs_;
	replaceBackend(nullptr);
	aiW_ = aiH_ = 0;
	PM_LOG_WARN("%d inference errors in a row; backend released, re-initialising in %.0f s", consecutiveErrors_,
		    reinitDelayMs_ / 1000.0);
	consecutiveErrors_ = 0;
	setState(WorkerState::Error, "inference failed repeatedly (" + error + "); retrying");
}

bool InferenceWorker::shouldLogError(double now)
{
	if (lastErrorLogMs_ < 0 || now - lastErrorLogMs_ >= kErrorLogIntervalMs) {
		lastErrorLogMs_ = now;
		return true;
	}
	++suppressedErrors_;
	return false;
}

void InferenceWorker::handleWorkerException(const char *what) noexcept
{
	try {
		const double now = Stopwatch::nowMs();
		if (shouldLogError(now)) {
			PM_LOG_ERROR("inference worker: unexpected exception: %s (%llu errors suppressed)", what,
				     suppressedErrors_);
			suppressedErrors_ = 0;
		}
		replaceBackend(nullptr);
		aiW_ = aiH_ = 0;
		consecutiveErrors_ = 0;
		// Same back-off as repeated inference errors; nothing to rebuild without a model.
		reinitDelayMs_ = reinitDelayMs_ <= 0 ? 1000.0 : std::min(reinitDelayMs_ * 2.0, kMaxReinitDelayMs);
		reinitDueMs_ = config_.modelPath.empty() ? -1 : now + reinitDelayMs_;
		setState(WorkerState::Error, std::string("internal error: ") + what);
	} catch (...) {
		// Nothing sensible left to do; the loop continues and waits for work.
	}
}

void InferenceWorker::tryReinitialize()
{
	reinitDueMs_ = -1;
	std::string err;
	std::unique_ptr<SegmentationBackend> fresh = initializeBackend(config_, err);
	if (!fresh) {
		reinitDelayMs_ = std::min(std::max(reinitDelayMs_, 1000.0) * 2.0, kMaxReinitDelayMs);
		reinitDueMs_ = Stopwatch::nowMs() + reinitDelayMs_;
		PM_LOG_ERROR("inference re-initialisation failed: %s; next attempt in %.0f s", err.c_str(),
			     reinitDelayMs_ / 1000.0);
		setState(WorkerState::Error, "re-initialisation failed: " + err);
		return;
	}
	if (!replaceBackend(std::move(fresh)))
		return; // stopping
	{
		std::lock_guard<std::mutex> lock(statusMutex_);
		status_.backend = backend_->capabilities();
	}
	PM_LOG_INFO("inference backend re-initialised (model '%s')", config_.model.id.c_str());
	updateAiResolution();
	setState(WorkerState::Running);
}

void InferenceWorker::reportCost(bool force)
{
	if (!backend_ || config_.model.id.empty())
		return;
	double cost = 0, budget = 0;
	int tier = 0;
	if (!perf_.measuredCost(cost, budget, tier))
		return;
	const double now = Stopwatch::nowMs();
	if (!force && lastCostReportMs_ >= 0 && now - lastCostReportMs_ < 5000)
		return;
	lastCostReportMs_ = now;
	CostObserver obs;
	{
		std::lock_guard<std::mutex> lock(observerMutex_);
		obs = costObserver_;
	}
	if (!obs)
		return;
	const auto caps = backend_->capabilities();
	callObserver("cost", [&] { obs(config_.model.id, caps.deviceName + "|" + caps.name, cost, budget, tier); });
}

void InferenceWorker::processOne(FrameBuffer &frame)
{
	if (frame.empty())
		return;
	Stopwatch total;
	// The throttle interval runs from the start of this frame.
	nextDueMs_ = config_.minIntervalMs > 0 ? Stopwatch::nowMs() + config_.minIntervalMs : 0.0;
	const ModelDescriptor &m = config_.model;

	// Fixed-input models must receive exactly their input size; if the render
	// thread has not caught up with a resolution change, resize on the CPU.
	const uint8_t *pixels = frame.bgra.data();
	uint32_t w = frame.width, h = frame.height, stride = frame.stride();
	uint32_t wantW = aiW_.load(), wantH = aiH_.load();
	if (wantW == 0 || wantH == 0)
		return;
	if (w != wantW || h != wantH) {
		resized_.resize(size_t(wantW) * wantH * 4);
		resizeBgra(pixels, w, h, stride, resized_.data(), wantW, wantH);
		pixels = resized_.data();
		w = wantW;
		h = wantH;
		stride = wantW * 4;
	}

	// Static scene: keep the previous matte instead of running the model. Not
	// recorded as an inference, so the performance controller never mistakes a
	// skipped frame for a fast one; RVM's recurrent state simply sees a longer
	// gap, as with a dropped frame.
	const double startMs = Stopwatch::nowMs();
	if (isStaticFrame(pixels, w, h, stride, startMs)) {
		logSkipStats(startMs);
		lastOutcome_ = MatteWait::Reused;
		return;
	}

	// Preprocess
	Stopwatch sw;
	const size_t n = size_t(w) * h;
	const bool fp16 = backend_->capabilities().fp16;
	PreprocessOptions po;
	po.layout = m.inputLayout;
	po.order = m.channelOrder;
	std::copy_n(m.mean, 3, po.mean);
	std::copy_n(m.std, 3, po.std);
	if (fp16) {
		tensorHalf_.resize(n * 3);
		bgraToTensorHalf(pixels, w, h, stride, po, tensorHalf_.data());
	} else {
		tensor_.resize(n * 3);
		bgraToTensor(pixels, w, h, stride, po, tensor_.data());
	}
	luma_.resize(n);
	bgraToLuma(pixels, w, h, stride, luma_.data());
	const double preMs = sw.elapsedMs();

	// Inference
	sw.reset();
	InferenceInput in;
	in.data = fp16 ? nullptr : tensor_.data();
	in.dataHalf = fp16 ? tensorHalf_.data() : nullptr;
	in.width = w;
	in.height = h;
	in.downsampleRatio = ratio_.load();
	in.shape = m.inputLayout == TensorLayout::NCHW ? std::vector<int64_t>{1, 3, int64_t(h), int64_t(w)}
						       : std::vector<int64_t>{1, int64_t(h), int64_t(w), 3};
	in.resetState = false;
	in.needForeground = needForeground_.load();
	std::string err;
	if (!backend_->processFrame(in, output_, err)) {
		if (running_)
			handleInferenceError(err);
		return;
	}
	const double infMs = sw.elapsedMs();
	consecutiveErrors_ = 0;
	reinitDelayMs_ = 0;
	// This frame is the new reference for the static-scene check.
	std::swap(thumb_, thumbRef_);
	thumbRefValid_ = thumbRef_.size() == size_t(kThumbW) * kThumbH;
	lastInferenceMs_ = startMs;
	logSkipStats(startMs);

	// Post-process: CPU refinement + temporal stabilisation at AI resolution.
	sw.reset();
	refiner_.process(output_.alpha.data(), w, h, config_.refine);
	double dt = lastProcessedMs_ >= 0 ? frame.timestampMs - lastProcessedMs_ : 33.3;
	temporal_.process(output_.alpha.data(), luma_.data(), w, h, dt, m.recurrent);
	lastProcessedMs_ = frame.timestampMs;

	matteWork_.resize(w, h);
	matteWork_.seq = frame.seq;
	matteWork_.hasForeground = !output_.foreground.empty();
	uint8_t *dst = matteWork_.rgba.data();
	if (matteWork_.hasForeground) {
		const float *r = output_.foreground.data();
		const float *g = r + n;
		const float *b = g + n;
		for (size_t i = 0; i < n; ++i) {
			dst[i * 4 + 0] = uint8_t(r[i] * 255.f + 0.5f);
			dst[i * 4 + 1] = uint8_t(g[i] * 255.f + 0.5f);
			dst[i * 4 + 2] = uint8_t(b[i] * 255.f + 0.5f);
			dst[i * 4 + 3] = uint8_t(std::clamp(output_.alpha[i], 0.f, 1.f) * 255.f + 0.5f);
		}
	} else {
		for (size_t i = 0; i < n; ++i) {
			dst[i * 4 + 0] = dst[i * 4 + 1] = dst[i * 4 + 2] = 0;
			dst[i * 4 + 3] = uint8_t(std::clamp(output_.alpha[i], 0.f, 1.f) * 255.f + 0.5f);
		}
	}
	const double postMs = sw.elapsedMs();

	// Developer diagnostics: mean input luma and mean alpha make it obvious when
	// the frame handed to the model is empty or the matte decoding is wrong.
	// (OBS only writes LOG_DEBUG when started with --verbose, so this goes to INFO.)
	if (log::minLevel() <= log::Level::Debug && (frame.seq % 30) == 0) {
		double lumaSum = 0, alphaSum = 0;
		for (size_t i = 0; i < n; ++i) {
			lumaSum += luma_[i];
			alphaSum += output_.alpha[i];
		}
		PM_LOG_INFO("frame %llu: %ux%u input luma %.1f, alpha mean %.3f (model %s)",
			     static_cast<unsigned long long>(frame.seq), w, h, lumaSum / double(n),
			     alphaSum / double(n), config_.model.id.c_str());
	}
	matteWork_.inferenceMs = infMs;
	matteWork_.totalMs = total.elapsedMs();

	{
		std::lock_guard<std::mutex> lock(matteMutex_);
		std::swap(matteSlot_, matteWork_);
		matteAvailable_ = true;
		publishedSeq_ = std::max(publishedSeq_, matteSlot_.seq);
	}
	lastOutcome_ = MatteWait::Ready; // waiters are woken by markFrameDone() in the loop
	if (settleFrames_ > 0) {
		--settleFrames_;
		perf_.recordSettlingInference();
	} else {
		perf_.recordInference(infMs, preMs, postMs, total.elapsedMs());
	}
	if (status_.state != WorkerState::Running)
		setState(WorkerState::Running);
}

bool InferenceWorker::isStaticFrame(const uint8_t *pixels, uint32_t w, uint32_t h, uint32_t stride, double nowMs)
{
	++skipWindowFrames_;
	if (!skipEnabled_.load()) {
		thumbRefValid_ = false;
		return false;
	}
	thumb_.resize(size_t(kThumbW) * kThumbH);
	lumaThumbnail(pixels, w, h, stride, kThumbW, kThumbH, thumb_.data());
	if (!thumbRefValid_ || thumbRef_.size() != thumb_.size() || lastInferenceMs_ < 0 ||
	    nowMs - lastInferenceMs_ >= skipMaxStaleMs_.load())
		return false;
	if (meanAbsDiff(thumb_.data(), thumbRef_.data(), thumb_.size()) >= double(skipThreshold_.load()))
		return false;
	++skipWindowSkipped_;
	return true;
}

void InferenceWorker::logSkipStats(double nowMs)
{
	if (skipWindowStartMs_ < 0) {
		skipWindowStartMs_ = nowMs;
		return;
	}
	if (nowMs - skipWindowStartMs_ < kSkipLogIntervalMs)
		return;
	if (skipEnabled_.load() && skipWindowFrames_ > 0)
		PM_LOG_INFO("static-scene skip: %llu of %llu frames reused the previous matte (%.0f%%) in the last %.0f s",
			    skipWindowSkipped_, skipWindowFrames_, 100.0 * double(skipWindowSkipped_) / double(skipWindowFrames_),
			    (nowMs - skipWindowStartMs_) / 1000.0);
	skipWindowStartMs_ = nowMs;
	skipWindowFrames_ = skipWindowSkipped_ = 0;
}

} // namespace promatte
