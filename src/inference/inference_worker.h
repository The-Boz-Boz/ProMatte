#pragma once
#include <atomic>
#include <functional>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "inference/frame_types.h"
#include "inference/segmentation_backend.h"
#include "performance/performance_controller.h"
#include "postprocessing/matte_refine.h"
#include "preprocessing/preprocess.h"
#include "temporal/temporal_stabilizer.h"

namespace promatte {

enum class WorkerState { Stopped, Idle, Initializing, Running, Error, NoModel };

// Outcome of InferenceWorker::waitForMatte().
enum class MatteWait {
	Ready,    // a matte for the frame (or a newer one) has been published
	Reused,   // the frame was a static-scene skip: the previous matte stays valid
	NoMatte,  // the frame was processed or dropped without a matte (error, disabled, no backend)
	Timeout,  // the frame is still queued or being processed
	Aborted,  // the worker is stopping or not Running
};

struct WorkerConfig {
	BackendConfig backend;
	ModelDescriptor model;
	std::string modelPath;
	// The same network in another precision (RVM FP32 for the FP16 file), or
	// empty. Tried when `model` cannot be initialised on a GPU backend, and
	// preferred over `model` on the CPU backend. When it is used the worker swaps
	// the two, so `model` always names what actually runs.
	ModelDescriptor altModel;
	std::string altModelPath;
	// Cheaper model used when even the lowest tier of `model` is too slow (Auto only).
	ModelDescriptor fallbackModel;
	std::string fallbackModelPath;
	// Better model tried when the current one has plenty of headroom (Auto only).
	ModelDescriptor upgradeModel;
	std::string upgradeModelPath;
	bool modelAutoSelected = false;
	QualityMode quality = QualityMode::Auto;
	int manualTier = -1; // >= 0 overrides quality mode (developer setting)
	uint32_t sourceWidth = 0;
	uint32_t sourceHeight = 0;
	double sourceFps = 30.0;
	TemporalParams temporal;
	RefineParams refine;
	double minIntervalMs = 0; // throttle: minimum time between inferences
};

struct WorkerStatus {
	WorkerState state = WorkerState::Stopped;
	std::string error;
	BackendCapabilities backend;
	std::string modelId;
	std::string modelName;
	uint32_t aiWidth = 0;
	uint32_t aiHeight = 0;
	int tier = 0;
	std::string tierLabel;
	float downsampleRatio = 1.f;
	bool recurrent = false;
	bool providesForeground = false;
	bool usingFallbackModel = false;
};

// Dedicated inference thread with a single-slot "latest frame wins" mailbox.
//
//   render thread ---submitFrame()---> [slot] ---> worker: preprocess, model,
//   refine, temporal ---> [matte slot] ---takeMatte()---> render thread
//
// By default the render thread never blocks: submitting overwrites an unread frame
// (counted as a drop), taking a matte only swaps buffers. In the synchronous mode
// the render thread additionally waits (bounded) for the matte of the frame it
// just submitted (waitForMatte). Backend initialisation, model loading and
// resolution changes all happen on the worker thread.
class InferenceWorker {
public:
	InferenceWorker();
	~InferenceWorker();

	InferenceWorker(const InferenceWorker &) = delete;
	InferenceWorker &operator=(const InferenceWorker &) = delete;

	void start();
	void stop(); // joins the thread; safe to call multiple times

	// Full (re)configuration; applied asynchronously by the worker thread.
	void setConfig(const WorkerConfig &config);
	// Hot parameters, applied on the next frame without re-initialisation.
	void setTuning(const TemporalParams &temporal, const RefineParams &refine);
	void setSourceInfo(uint32_t width, uint32_t height, double fps);
	void setQualityMode(QualityMode mode, int manualTier);
	// Static-scene skip: a frame whose luma thumbnail shows no local motion
	// against the last inferred frame (measureMotion with `params`) is not run
	// through the model, as long as the last real inference is younger than
	// `maxStaleMs`. The previous matte stays on screen. Comparing with the last
	// inferred frame, not the previous one, lets slow drift accumulate until it
	// counts as motion. Hot parameter.
	void setMotionSkip(bool enabled, const MotionSkipParams &params, double maxStaleMs);
	// Whether mattes need the model's foreground colour (RVM fgr). Hot parameter.
	void setNeedForeground(bool need) { needForeground_ = need; }
	// Whether the render thread waits for each matte (synchronous mode). Only
	// affects the worker thread priority (threadPriorityNormal). Hot parameter.
	void setSyncMatte(bool sync) { syncMatte_ = sync; }
	// Worker thread priority policy: normal while OBS' graphics thread waits for
	// this thread (synchronous matte) and the work runs on a GPU, so the CPU side
	// of a dispatch is not queued behind other normal-priority threads; below
	// normal otherwise, so CPU inference cannot starve the compositor/encoders.
	static bool threadPriorityNormal(bool syncMatte, bool gpuBackend) { return syncMatte && gpuBackend; }
	// When disabled the worker drops incoming frames and idles (no AI cost).
	void setEnabled(bool enabled);
	// Release the backend/session (e.g. filter hidden for a long time).
	void releaseBackend();

	// --- render-thread side -------------------------------------------------
	// AI resolution the render thread should downscale to right now.
	void desiredResolution(uint32_t &w, uint32_t &h) const;
	// True when the worker has consumed the previous frame, is ready for a new one
	// and the maxAiFps throttle allows starting it. The render thread uses this to
	// avoid a GPU readback (a sync point that can queue behind long inference
	// dispatches) for frames nobody will process.
	bool wantsFrame() const;
	// Hands a frame to the worker; `frame` receives the previous slot buffer for reuse.
	// Returns the sequence number assigned to the submitted frame.
	uint64_t submitFrame(FrameBuffer &frame);
	// Blocks until the frame `seq` has been handled by the worker or `timeoutMs`
	// elapses. Returns early (Aborted) when the worker stops or leaves Running.
	// Never spins: waits on a condition variable signalled by the worker.
	MatteWait waitForMatte(uint64_t seq, double timeoutMs);
	// Swaps in the newest matte if it is newer than `out.seq`. Returns true if updated.
	bool takeMatte(MatteBuffer &out);
	bool hasMatte() const { return matteAvailable_.load(); }

	WorkerStatus status() const;
	// Cheap state query for the render thread (no string copies).
	WorkerState state() const;
	PerfStats stats() const;
	PerformanceController &performance() { return perf_; }
	void resetTemporal();

	// Called (from the worker thread) when a model's cost has been measured, so the
	// caller can persist it: (modelId, deviceKey, costMs, budgetMs, tier).
	using CostObserver = std::function<void(const std::string &, const std::string &, double, double, int)>;
	void setCostObserver(CostObserver observer);
	// Called when the worker switched models by itself; the caller may re-read status().
	using ModelChangeObserver = std::function<void(const std::string &)>;
	void setModelChangeObserver(ModelChangeObserver observer);
	// Called (worker thread) when GPU inference keeps starving OBS rendering even at
	// the cheapest configuration; the owner should reconfigure with the CPU backend.
	using BackendFallbackObserver = std::function<void(const std::string &deviceKey)>;
	void setBackendFallbackObserver(BackendFallbackObserver observer);

private:
	void threadMain();
	// Applies threadPriorityNormal() for the current mode and backend (worker
	// thread; a no-op when nothing changed and on non-Windows platforms).
	void updateThreadPriority();
	// keepCurrentOnFailure: used for automatic model switches - when the new model
	// cannot be initialised, the current model and backend stay in service.
	bool applyConfigLocked(WorkerConfig cfg, bool keepCurrentOnFailure = false);
	// Creates and initialises a backend for `cfg` without touching backend_.
	// Swaps cfg.model and cfg.altModel when the alternative precision is the one
	// that initialised.
	std::unique_ptr<SegmentationBackend> initializeBackend(WorkerConfig &cfg, std::string &error);
	void publishModel(const WorkerConfig &cfg, bool usingFallback);
	// The only way backend_ changes (worker thread, or stop() after the join).
	// Refuses to install a new backend while the worker is being stopped.
	bool replaceBackend(std::unique_ptr<SegmentationBackend> next);
	void handleInferenceError(const std::string &error);
	// Last line of defence for the worker loop: log, drop the backend and recover
	// through the regular back-off instead of letting the thread die.
	void handleWorkerException(const char *what) noexcept;
	// Error log rate limit shared by inference errors and worker exceptions.
	bool shouldLogError(double now);
	void tryReinitialize();
	void updateAiResolution();
	void processOne(FrameBuffer &frame);
	// True when `pixels` is close enough to the last inferred frame to reuse its
	// matte. Leaves the frame's thumbnail in thumb_ for processOne.
	bool isStaticFrame(const uint8_t *pixels, uint32_t w, uint32_t h, uint32_t stride, double nowMs);
	void logSkipStats(double nowMs);
	void reportCost(bool force);
	void setState(WorkerState s, const std::string &error = {});
	// Records that frame `seq` left the worker (processed, skipped or dropped) and
	// wakes waitForMatte().
	void markFrameDone(uint64_t seq, MatteWait outcome);
	void wakeMatteWaiters();

	// thread + control
	std::thread thread_;
	std::atomic<bool> running_{false};
	std::atomic<bool> stopping_{false};
	std::atomic<bool> enabled_{true};
	mutable std::mutex ctrlMutex_;
	std::condition_variable cv_;
	bool configPending_ = false;
	bool releasePending_ = false;
	WorkerConfig pendingConfig_;
	bool tuningPending_ = false;
	TemporalParams pendingTemporal_;
	RefineParams pendingRefine_;
	bool sourcePending_ = false;
	uint32_t pendingSrcW_ = 0, pendingSrcH_ = 0;
	double pendingSrcFps_ = 30;
	bool qualityPending_ = false;
	QualityMode pendingQuality_ = QualityMode::Auto;
	int pendingManualTier_ = -1;
	bool resetTemporalPending_ = false;

	// frame mailbox
	mutable std::mutex frameMutex_;
	FrameBuffer frameSlot_;
	bool frameAvailable_ = false;
	uint64_t nextSeq_ = 1;

	// matte mailbox
	mutable std::mutex matteMutex_;
	MatteBuffer matteSlot_;
	std::atomic<bool> matteAvailable_{false};
	// Synchronous mode bookkeeping, guarded by matteMutex_.
	std::condition_variable matteCv_;
	uint64_t publishedSeq_ = 0; // seq of the newest published matte
	uint64_t doneSeq_ = 0;      // newest seq the worker finished with (any outcome)
	MatteWait doneOutcome_ = MatteWait::NoMatte;
	// Mirror of status_.state readable without statusMutex_.
	std::atomic<WorkerState> stateFast_{WorkerState::Stopped};
	// Outcome of the last processOne() (worker thread only).
	MatteWait lastOutcome_ = MatteWait::NoMatte;

	// worker-owned state
	WorkerConfig config_;
	// Owned by the worker thread, which may read it without locking. Every
	// assignment/reset holds backendMutex_, and so does stop() while it calls
	// cancel() from another thread. Run/initialise never hold it.
	std::mutex backendMutex_;
	std::unique_ptr<SegmentationBackend> backend_;
	std::vector<float> tensor_;
	std::vector<uint16_t> tensorHalf_; // float16 models (BackendCapabilities::fp16)
	std::vector<uint8_t> luma_;
	std::vector<uint8_t> resized_;
	FrameBuffer work_;
	InferenceOutput output_;
	MatteBuffer matteWork_;
	MatteRefiner refiner_;
	TemporalStabilizer temporal_;
	PerformanceController perf_;
	double lastProcessedMs_ = -1;
	int tier_ = 0;
	// Inferences after an AI resolution change that are not timed for the
	// performance controller (kSettleFrames, set by updateAiResolution).
	static constexpr int kSettleFrames = 2;
	int settleFrames_ = 0;

	// static-scene skip (setMotionSkip)
	static constexpr uint32_t kThumbW = 64, kThumbH = 36;
	static constexpr double kSkipLogIntervalMs = 60000.0;
	std::atomic<bool> needForeground_{true};
	std::atomic<bool> syncMatte_{false};
	int threadPriorityNormal_ = -1; // last applied policy: -1 unknown, 0 below normal, 1 normal
	std::atomic<bool> skipEnabled_{true};
	std::atomic<float> skipCellDelta_{MotionSkipParams{}.cellDelta};
	std::atomic<int> skipMinCells_{MotionSkipParams{}.minChangedCells};
	std::atomic<double> skipMaxStaleMs_{150.0};
	std::vector<uint8_t> thumb_, thumbRef_; // current frame / last inferred frame
	bool thumbRefValid_ = false;            // cleared whenever the matte stream restarts
	double lastInferenceMs_ = -1;
	double skipWindowStartMs_ = -1;
	unsigned long long skipWindowFrames_ = 0, skipWindowSkipped_ = 0;
	unsigned long long skipWindowStale_ = 0; // frames that looked static but hit maxStaleMs
	double skipWindowNoiseSum_ = 0;          // sum of measured noise floors, for the log
	unsigned long long skipWindowMeasured_ = 0;

	// error recovery: after kErrorsBeforeReset failed runs in a row the backend is
	// dropped and re-initialised after 1, 2, 4, ... kMaxReinitDelayMs.
	static constexpr int kErrorsBeforeReset = 3;
	static constexpr double kErrorLogIntervalMs = 10000.0;
	static constexpr double kMaxReinitDelayMs = 30000.0;
	int consecutiveErrors_ = 0;
	unsigned long long suppressedErrors_ = 0;
	double lastErrorLogMs_ = -1;
	double reinitDelayMs_ = 0;
	double reinitDueMs_ = -1; // < 0: no re-initialisation scheduled
	std::atomic<uint32_t> aiW_{0}, aiH_{0};
	// Throttle (minIntervalMs): earliest Stopwatch::nowMs() at which the next frame
	// may start, measured from the start of the previous one. 0 = no limit.
	std::atomic<double> nextDueMs_{0.0};
	std::atomic<float> ratio_{1.f};

	// status
	mutable std::mutex statusMutex_;
	WorkerStatus status_;
	mutable std::mutex observerMutex_;
	CostObserver costObserver_;
	ModelChangeObserver modelChangeObserver_;
	BackendFallbackObserver backendFallbackObserver_;
	double lastCostReportMs_ = -1;
};

const char *workerStateName(WorkerState s);

} // namespace promatte
