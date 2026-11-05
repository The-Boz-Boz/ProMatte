#include "obs/filter.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>

#include <util/platform.h>

#include "inference/backend_registry.h"
#include "models/model_manager.h"
#include "obs/presets.h"
#include "utils/logging.h"
#include "utils/system_info.h"
#include "utils/timer.h"

namespace promatte {

obs_properties_t *buildProperties(void *data); // ui/properties.cpp

namespace {

const char *filterName(void *)
{
	return obs_module_text("FilterName");
}

double sourceFps()
{
	struct obs_video_info ovi;
	if (obs_get_video_info(&ovi) && ovi.fps_den)
		return double(ovi.fps_num) / double(ovi.fps_den);
	return 30.0;
}

// Model ids this device is known to be unable to run in real time.
std::function<bool(const std::string &)> blockedModels(const std::string &deviceKey)
{
	return [deviceKey](const std::string &id) { return capabilityStore().isTooSlow(deviceKey, id); };
}

std::string resolveModelId(const FilterSettings &s, bool gpu, const std::string &deviceKey)
{
	ModelManager &mm = modelManager();
	auto blocked = blockedModels(deviceKey);
	if (s.modelId != "auto") {
		// "rvm_mobilenetv3" stands for the network, not the file: GPU backends get
		// the FP16 export when it is installed, the CPU gets FP32. Saved settings
		// keep naming the base id, so nothing has to be migrated.
		const std::string id = mm.preferredVariant(s.modelId, gpu, blocked);
		if (!id.empty() && mm.isInstalled(id))
			return id;
		PM_LOG_WARN("model '%s' is not installed; falling back to automatic selection", s.modelId.c_str());
	}
	// Prefer the best model this device has already been measured to run inside its
	// budget; otherwise start with the cheapest one and let the controller climb.
	const CapabilityStore &store = capabilityStore();
	const ModelDescriptor *bestKnown = nullptr;
	int bestQuality = -1;
	std::vector<ModelStatus> all = mm.list();
	for (const auto &st : all) {
		if (!st.installed || st.desc.family == "custom" || blocked(st.desc.id))
			continue;
		auto rec = store.get(deviceKey, st.desc.id);
		if (rec.updated == 0 || rec.tooSlow || rec.budgetMs <= 0 || rec.costMs > rec.budgetMs)
			continue;
		if (st.desc.qualityRank > bestQuality) {
			bestQuality = st.desc.qualityRank;
			bestKnown = &st.desc;
		}
	}
	if (bestKnown)
		return mm.preferredVariant(bestKnown->id, gpu, blocked);
	auto d = mm.selectModel(gpu, /*preferCheap=*/true, blocked);
	return d ? mm.preferredVariant(d->id, gpu, blocked) : std::string{};
}

} // namespace

std::string FilterContext::statusText(bool multiline) const
{
	if (!worker)
		return "Inference worker not running";
	WorkerStatus st = worker->status();
	PerfStats ps = worker->stats();
	const char *nl = multiline ? "\n" : "  |  ";
	char buf[1024];
	std::string out;
	switch (st.state) {
	case WorkerState::Stopped:
	case WorkerState::Idle:
		return std::string("AI processing idle") + (gpuReady ? "" : " (GPU pipeline unavailable: " + gpuError + ")");
	case WorkerState::Initializing:
		return "Loading model '" + st.modelName + "'...";
	case WorkerState::NoModel:
		return "No AI model installed - open the Model manager section below and download a model. The filter passes video through unchanged.";
	case WorkerState::Error:
		return "AI processing error: " + st.error + " (video passes through unchanged)";
	case WorkerState::Running:
		break;
	}
	std::snprintf(buf, sizeof(buf), "Backend: %s (%s)%sModel: %s%sAI: %ux%u (%s)%s", st.backend.name.c_str(),
		      st.backend.deviceName.c_str(), nl, st.modelName.c_str(), nl, st.aiWidth, st.aiHeight,
		      st.tierLabel.c_str(), nl);
	out += buf;
	std::snprintf(buf, sizeof(buf), "AI processing: %.0f FPS, inference %.1f ms (total %.1f ms)%sRender: %.2f ms%sDropped: %.1f%% (%llu of %llu)",
		      ps.aiFps, ps.inferenceMs, ps.totalMs, nl, ps.renderMs, nl, ps.dropRatio * 100.0,
		      static_cast<unsigned long long>(ps.framesDropped),
		      static_cast<unsigned long long>(ps.framesSubmitted));
	out += buf;
	if (!ps.message.empty())
		out += std::string(nl) + ps.message;
	return out;
}

void FilterContext::configureWorker(const std::string &modelOverride)
{
	if (!worker)
		return;
	FilterSettings s;
	{
		std::lock_guard<std::mutex> lock(settingsMutex);
		s = settings;
	}
	WorkerConfig cfg;
	cfg.backend.kind = backendKindFromString(s.backend);
	cfg.backend.cpuThreads = s.cpuThreads;
	cfg.backend.allowFallback = true;
	cfg.backend.deviceIndex = s.gpuIndex >= 0 ? s.gpuIndex : preferredGpuAdapter(renderAdapterName);
	BackendKind effective = cfg.backend.kind == BackendKind::Auto ? resolveAutoBackend() : cfg.backend.kind;
	// Auto backend: skip a GPU that is known (this session or a previous one) to
	// starve OBS rendering on this machine.
	if (cfg.backend.kind == BackendKind::Auto && effective != BackendKind::CPU) {
		std::string gpuName;
		for (const auto &c : enumerateBackends())
			if (c.kind == effective && !c.deviceName.empty())
				gpuName = c.deviceName;
		const std::string gpuKey = gpuName + "|" + backendKindName(effective);
		if (forceCpuBackend.load() || capabilityStore().isTooSlow(gpuKey, "__backend__")) {
			PM_LOG_INFO("auto backend: %s is known to contend with OBS rendering on this machine; using CPU",
				    gpuKey.c_str());
			effective = BackendKind::CPU;
			cfg.backend.kind = BackendKind::CPU;
		}
	}
	const bool gpuBackend = effective != BackendKind::CPU && backendAvailable(effective);
	// The device key must match what the worker reports (backend device name + backend).
	std::string deviceName = renderAdapterName;
	for (const auto &c : enumerateBackends())
		if (c.kind == effective && !c.deviceName.empty())
			deviceName = c.deviceName;
	if (!gpuBackend)
		deviceName = sysinfo::cpuName();
	deviceKey = deviceName + "|" + backendKindName(effective);
	ModelManager &mm = modelManager();
	// The worker switched models on its own (Auto): keep its choice and only
	// refresh the fallback/upgrade candidates around it.
	std::string modelId = !modelOverride.empty() && s.modelId == "auto" && mm.isInstalled(modelOverride)
				      ? modelOverride
				      : resolveModelId(s, gpuBackend, deviceKey);
	if (!modelId.empty()) {
		auto d = mm.find(modelId);
		if (d) {
			cfg.model = *d;
			cfg.modelPath = mm.resolvePath(modelId);
		}
		// The other precision of the same network: the worker falls back to it
		// when the FP16 file fails to initialise on the GPU (and prefers it on CPU).
		const std::string altId = mm.alternatePrecision(modelId);
		if (!altId.empty()) {
			if (auto a = mm.find(altId)) {
				cfg.altModel = *a;
				cfg.altModelPath = mm.resolvePath(altId);
			}
		}
	}
	resolvedModelId = modelId;
	configuredBackend = backendKindName(effective);
	cfg.modelAutoSelected = s.modelId == "auto";
	if (cfg.modelAutoSelected) {
		auto blocked = blockedModels(deviceKey);
		// Cheapest usable model as the emergency fallback.
		auto pick = [&](const std::string &id, ModelDescriptor &desc, std::string &path) {
			const std::string variant = mm.preferredVariant(id, gpuBackend, blocked);
			if (auto d = mm.find(variant)) {
				desc = *d;
				path = mm.resolvePath(variant);
			}
		};
		const std::string baseId = cfg.model.variantOf.empty() ? modelId : cfg.model.variantOf;
		if (auto fb = mm.selectModel(false, /*preferCheap=*/true, [&](const std::string &id) {
			    return id == modelId || id == baseId || blocked(id);
		    }))
			pick(fb->id, cfg.fallbackModel, cfg.fallbackModelPath);
		// Next model up the quality ladder, tried when there is measured headroom.
		if (auto up = mm.betterModel(modelId, gpuBackend, blocked))
			pick(up->id, cfg.upgradeModel, cfg.upgradeModelPath);
	}
	cfg.quality = s.quality;
	cfg.manualTier = s.manualTier;
	cfg.sourceWidth = lastW;
	cfg.sourceHeight = lastH;
	cfg.sourceFps = lastFps > 1 ? lastFps : sourceFps();
	cfg.temporal = s.temporalParams();
	cfg.refine = s.refineParams();
	cfg.minIntervalMs = s.maxAiFps > 0 ? 1000.0 / double(s.maxAiFps) : 0.0;
	worker->setMotionSkip(s.motionSkip, s.motionSkipParams(), double(s.motionSkipMaxStaleMs));
	worker->setNeedForeground(s.needsModelForeground());
	worker->setConfig(cfg);
	PM_LOG_INFO("filter '%s': backend=%s model=%s quality=%s", obs_source_get_name(source), configuredBackend.c_str(),
		    modelId.empty() ? "(none)" : modelId.c_str(), qualityModeName(s.quality));
}

void FilterContext::applyTuning()
{
	if (!worker)
		return;
	std::lock_guard<std::mutex> lock(settingsMutex);
	worker->setTuning(settings.temporalParams(), settings.refineParams());
	worker->setQualityMode(settings.quality, settings.manualTier);
	worker->setMotionSkip(settings.motionSkip, settings.motionSkipParams(), double(settings.motionSkipMaxStaleMs));
	worker->setNeedForeground(settings.needsModelForeground());
}

namespace {

// libobs is C: an exception must never unwind into it. The obs_source_info
// callbacks below are thin wrappers that log and swallow. `lastLogNs` limits
// per-frame callbacks (tick/render) to one log line per 10 s.
void reportCallbackException(const char *where, const char *what, std::atomic<uint64_t> *lastLogNs = nullptr)
{
	if (lastLogNs) {
		const uint64_t now = os_gettime_ns();
		const uint64_t last = lastLogNs->load();
		if (last && now - last < 10000000000ULL)
			return;
		lastLogNs->store(now);
	}
	PM_LOG_ERROR("%s: unexpected exception: %s", where, what);
}

template<typename Fn> bool guarded(const char *where, Fn &&fn, std::atomic<uint64_t> *lastLogNs = nullptr) noexcept
{
	try {
		fn();
		return true;
	} catch (const std::exception &e) {
		reportCallbackException(where, e.what(), lastLogNs);
	} catch (...) {
		reportCallbackException(where, "unknown exception", lastLogNs);
	}
	return false;
}

void filterDestroy(void *data);

void filterCreateImpl(FilterContext *ctx, obs_data_t *data, obs_source_t *source)
{
	ctx->source = source;
	ctx->settings = settingsLoad(data);

	obs_enter_graphics();
	std::string effectsDir, err;
	const char *dir = moduleDataPath("effects", effectsDir);
	ctx->gpuReady = dir && ctx->gpu.loadEffects(dir, err);
	if (const char *adapter = gs_get_device_name())
		ctx->renderAdapterName = adapter;
	obs_leave_graphics();
	if (!ctx->gpuReady) {
		ctx->gpuError = err.empty() ? "effects directory missing" : err;
		PM_LOG_ERROR("GPU pipeline unavailable: %s", ctx->gpuError.c_str());
	}

	ctx->worker = std::make_unique<InferenceWorker>();
	// Persist measured model costs so "Auto" does not relearn on every start.
	ctx->worker->setCostObserver([](const std::string &modelId, const std::string &device, double costMs,
					double budgetMs, int tier) {
		if (tier == 0) // only the cheapest tier is comparable across runs
			capabilityStore().record(device, modelId, costMs, budgetMs);
	});
	ctx->worker->setModelChangeObserver([ctx](const std::string &) { ctx->modelChangedByWorker = true; });
	// GPU inference starving the compositor: remember it and move to the CPU.
	ctx->worker->setBackendFallbackObserver([ctx](const std::string &deviceKey) {
		for (int i = 0; i < CapabilityStore::kSlowStrikes; ++i)
			capabilityStore().record(deviceKey, "__backend__", 1e6, 1.0);
		ctx->forceCpuBackend = true;
		ctx->reconfigureRequested = true; // applied by filterTick on the graphics thread
	});
	ctx->worker->start();
	ctx->lastFps = sourceFps();
	ctx->configureWorker();
	ctx->background.update(ctx->settings, source);
	PM_LOG_INFO("ProMatte filter created on '%s'", obs_source_get_name(source));
}

void *filterCreate(obs_data_t *data, obs_source_t *source)
{
	FilterContext *ctx = nullptr;
	if (guarded("filter_create", [&] {
		    ctx = new FilterContext();
		    filterCreateImpl(ctx, data, source);
	    }))
		return ctx;
	// Half-built context: tear down whatever was set up (stops the worker).
	if (ctx)
		filterDestroy(ctx);
	return nullptr;
}

void filterDestroyImpl(void *data)
{
	auto *ctx = static_cast<FilterContext *>(data);
	if (!ctx)
		return;
	// Stop the worker first (joins the thread; cancels an in-flight run).
	if (ctx->worker)
		ctx->worker->stop();
	obs_enter_graphics();
	ctx->background.release();
	ctx->overlay.release();
	ctx->gpu.free();
	obs_leave_graphics();
	ctx->worker.reset();
	PM_LOG_INFO("ProMatte filter destroyed");
	delete ctx;
}

void filterDestroy(void *data)
{
	guarded("filter_destroy", [&] { filterDestroyImpl(data); });
}

void filterUpdateImpl(void *data, obs_data_t *settings)
{
	auto *ctx = static_cast<FilterContext *>(data);
	FilterSettings fresh = settingsLoad(settings);
	FilterSettings old;
	{
		std::lock_guard<std::mutex> lock(ctx->settingsMutex);
		old = ctx->settings;
		ctx->settings = fresh;
	}
	ctx->background.update(fresh, ctx->source);
	const bool reconfigure = fresh.modelId != old.modelId || fresh.backend != old.backend ||
				 fresh.cpuThreads != old.cpuThreads || fresh.maxAiFps != old.maxAiFps ||
				 fresh.gpuIndex != old.gpuIndex ||
				 (ctx->worker && ctx->worker->status().state == WorkerState::NoModel);
	if (reconfigure)
		ctx->reconfigureRequested = true; // applied by filterTick on the graphics thread
	else
		ctx->applyTuning();
}

void filterUpdate(void *data, obs_data_t *settings)
{
	guarded("filter_update", [&] { filterUpdateImpl(data, settings); });
}

void filterDefaults(obs_data_t *settings)
{
	settingsDefaults(settings);
}

obs_properties_t *filterProperties(void *data)
{
	obs_properties_t *props = nullptr;
	if (guarded("filter_properties", [&] { props = buildProperties(data); }) && props)
		return props;
	return obs_properties_create(); // never hand OBS a null property list
}

void filterTickImpl(void *data, float seconds)
{
	auto *ctx = static_cast<FilterContext *>(data);
	ctx->background.tick(seconds);
	// All (re)configuration happens here, on the graphics thread: requests from the
	// UI, filter_update and the worker thread only raise flags.
	// A model that finished downloading may be the better file for this
	// selection (e.g. RVM FP16 for "rvm_mobilenetv3" on DirectML): re-resolve.
	{
		const uint64_t nowNs = os_gettime_ns();
		if (nowNs - ctx->lastDownloadCheckNs >= 1000000000ULL) {
			ctx->lastDownloadCheckNs = nowNs;
			const DownloadState dl = modelManager().downloadState();
			if (ctx->downloadWasActive && !dl.active && dl.succeeded)
				ctx->reconfigureRequested = true;
			ctx->downloadWasActive = dl.active;
		}
	}
	const bool modelChanged = ctx->modelChangedByWorker.exchange(false);
	if (ctx->reconfigureRequested.exchange(false))
		ctx->configureWorker();
	else if (modelChanged && ctx->worker)
		ctx->configureWorker(ctx->worker->status().modelId);
	// Requirement E: when the filter is disabled OBS stops calling video_render;
	// detect that and idle the worker so no AI work happens.
	const uint64_t now = os_gettime_ns();
	const bool rendering = ctx->lastRenderNs && (now - ctx->lastRenderNs) < 400000000ULL; // 400 ms
	if (ctx->worker && ctx->workerEnabled.load() != rendering) {
		ctx->workerEnabled = rendering;
		ctx->worker->setEnabled(rendering);
		PM_LOG_DEBUG("filter %s -> worker %s", rendering ? "rendering" : "idle", rendering ? "enabled" : "paused");
	}
}

void filterTick(void *data, float seconds)
{
	static std::atomic<uint64_t> lastLogNs{0};
	guarded("filter_tick", [&] { filterTickImpl(data, seconds); }, &lastLogNs);
}

void updateOverlay(FilterContext *ctx, uint32_t w, uint32_t h)
{
	const uint64_t now = os_gettime_ns();
	if (now - ctx->lastOverlayNs < 500000000ULL)
		return;
	ctx->lastOverlayNs = now;
	WorkerStatus st = ctx->worker->status();
	PerfStats ps = ctx->worker->stats();
	char buf[1024];
	std::snprintf(buf, sizeof(buf),
		      "ProMatte %s\nBackend: %s (%s)\nModel: %s\nAI: %ux%u %s (source %ux%u)\nInference: %.1f ms  (pre %.2f, post %.2f)\n"
		      "Processing FPS: %.1f   Render: %.2f ms\nDropped: %.1f%%   Queue: %d   State: %s\nVRAM (process): %.0f MB   RSS: %.0f MB",
		      PROMATTE_VERSION, st.backend.name.c_str(), st.backend.deviceName.c_str(), st.modelName.c_str(),
		      st.aiWidth, st.aiHeight, st.tierLabel.c_str(), w, h, ps.inferenceMs, ps.preprocessMs,
		      ps.postprocessMs, ps.aiFps, ps.renderMs, ps.dropRatio * 100.0, ps.queueDepth, workerStateName(st.state),
		      double(sysinfo::processVideoMemoryBytes()) / (1024.0 * 1024.0),
		      double(sysinfo::processWorkingSetBytes()) / (1024.0 * 1024.0));
	std::string text = buf;
	if (!ps.message.empty())
		text += "\n" + ps.message;
	if (st.state == WorkerState::Error)
		text += "\nError: " + st.error;
	ctx->overlay.setText(text);
}

// Synchronous matte: waits (bounded) for the matte of frame `seq` (0 = nothing
// was submitted this frame) and keeps the statistics logged once a minute.
void syncMatteWait(FilterContext *ctx, uint64_t seq, int timeoutSettingMs)
{
	FilterContext::SyncStats &st = ctx->sync;
	const uint64_t now = os_gettime_ns();
	if (!st.windowStartNs)
		st.windowStartNs = now;
	++st.frames;
	if (seq) {
		const double fps = ctx->lastFps > 1 ? ctx->lastFps : sourceFps();
		const double timeoutMs = std::min(double(timeoutSettingMs), 0.9 * 1000.0 / fps);
		Stopwatch sw;
		const MatteWait r = ctx->worker->waitForMatte(seq, timeoutMs);
		const double waitedMs = sw.elapsedMs();
		++st.waited;
		st.waitSumMs += waitedMs;
		st.waitMaxMs = std::max(st.waitMaxMs, waitedMs);
		switch (r) {
		case MatteWait::Ready:
			++st.synced;
			break;
		case MatteWait::Reused:
			++st.reused;
			break;
		case MatteWait::Timeout:
			++st.timeouts;
			break;
		case MatteWait::NoMatte:
		case MatteWait::Aborted:
			++st.other;
			break;
		}
	}
	if (now - st.windowStartNs >= 60000000000ULL) {
		if (st.frames > 0)
			PM_LOG_INFO("sync matte: %llu frames, %llu waited: %llu in sync (%.0f%% of frames), %llu static reuse, "
				    "%llu timeouts, %llu without matte; wait avg %.1f ms, max %.1f ms",
				    st.frames, st.waited, st.synced, 100.0 * double(st.synced) / double(st.frames),
				    st.reused, st.timeouts, st.other,
				    st.waited ? st.waitSumMs / double(st.waited) : 0.0, st.waitMaxMs);
		st = FilterContext::SyncStats{};
		st.windowStartNs = now;
	}
}

void filterRenderImpl(void *data)
{
	auto *ctx = static_cast<FilterContext *>(data);
	obs_source_t *target = obs_filter_get_target(ctx->source);
	obs_source_t *parent = obs_filter_get_parent(ctx->source);
	const uint32_t w = target ? obs_source_get_base_width(target) : 0;
	const uint32_t h = target ? obs_source_get_base_height(target) : 0;
	ctx->lastRenderNs = os_gettime_ns();
	// OBS renders each filter once for the program output and again for every
	// preview/projector. Only the first render of a video frame does the AI feed
	// and the expensive refinement passes; the rest reuse them.
	const uint64_t frameTime = obs_get_video_frame_time();
	const bool firstRenderOfFrame = frameTime != ctx->lastFrameTime;
	ctx->lastFrameTime = frameTime;

	// Fail-safe: anything missing -> draw the source untouched.
	auto passthrough = [&] {
		obs_source_skip_video_filter(ctx->source);
	};
	if (!ctx->gpuReady || !ctx->worker || w == 0 || h == 0 || !parent) {
		passthrough();
		return;
	}
	Stopwatch sw;

	FilterSettings s;
	{
		std::lock_guard<std::mutex> lock(ctx->settingsMutex);
		s = ctx->settings;
	}

	// Track source size / fps changes.
	if (w != ctx->lastW || h != ctx->lastH) {
		ctx->lastW = w;
		ctx->lastH = h;
		ctx->lastFps = sourceFps();
		ctx->worker->setSourceInfo(w, h, ctx->lastFps);
		PM_LOG_INFO("source size %ux%u @ %.2f fps", w, h, ctx->lastFps);
	}

	// While the worker is loading, failed or has no model it produces no new
	// mattes; drawing the last one would freeze the cut-out over live video.
	// Unless a fresh matte is already waiting, show the source untouched and skip
	// the full-resolution capture.
	const WorkerState workerState = ctx->worker->state();
	if ((workerState == WorkerState::Error || workerState == WorkerState::Initializing ||
	     workerState == WorkerState::NoModel) &&
	    !ctx->worker->hasMatte()) {
		if (ctx->gpu.hasMatte())
			ctx->gpu.dropMatte();
		passthrough();
		return;
	}

	// 1. Capture the source into our texture.
	if (!ctx->gpu.captureBegin(w, h)) {
		passthrough();
		return;
	}
	if (obs_source_process_filter_begin(ctx->source, GS_RGBA, OBS_NO_DIRECT_RENDERING))
		obs_source_process_filter_end(ctx->source, obs_get_base_effect(OBS_EFFECT_DEFAULT), w, h);
	ctx->gpu.captureEnd();

	// 2. Feed the inference worker.
	//    The downscale + stage is an asynchronous GPU copy and runs every frame so
	//    the staged frame stays fresh. The readback (gs_stagesurface_map) is a GPU
	//    sync point that can queue behind long inference dispatches, so it only
	//    happens when the worker is actually ready to take a frame.
	//
	//    Asynchronous mode (sync off): the frame read back is the one staged on
	//    the previous render, and its matte is drawn on a later frame (never
	//    blocks, but the matte trails the video by about two frames).
	//    Synchronous mode: stage this frame, read it back at once (the map waits
	//    for the copy), hand it to the worker and wait - bounded - for its matte,
	//    so the matte is composited onto the frame it was computed from.
	uint32_t aiW = 0, aiH = 0;
	ctx->worker->desiredResolution(aiW, aiH);
	if (firstRenderOfFrame && aiW && aiH && ctx->workerEnabled.load()) {
		if (s.syncMatte) {
			ctx->gpu.stageFrame(aiW, aiH);
			const bool running = workerState == WorkerState::Running;
			uint64_t seq = 0;
			// wantsFrame() is false while maxAiFps throttles or an earlier frame
			// still sits unread in the slot: then nothing is fed and nothing is
			// waited for (the newest available matte is drawn, as in async mode).
			if (running && ctx->worker->wantsFrame() && ctx->gpu.readbackFrame(ctx->frame))
				seq = ctx->worker->submitFrame(ctx->frame);
			syncMatteWait(ctx, seq, s.syncMatteTimeoutMs);
		} else {
			if (ctx->worker->wantsFrame() && ctx->gpu.readbackFrame(ctx->frame))
				ctx->worker->submitFrame(ctx->frame);
			ctx->gpu.stageFrame(aiW, aiH);
		}
	}

	// 3. Pick up the newest matte.
	if (firstRenderOfFrame && ctx->worker->takeMatte(ctx->matte))
		ctx->gpu.uploadMatte(ctx->matte);

	// 4. Compose.
	if (!ctx->gpu.hasMatte()) {
		// No matte yet (model loading / error): show the original video.
		ctx->gpu.renderPassthrough();
	} else {
		RenderParams p;
		p.mode = s.mode;
		p.fit = s.fit;
		struct vec4 c;
		vec4_from_rgba(&c, s.color);
		p.color[0] = c.x;
		p.color[1] = c.y;
		p.color[2] = c.z;
		p.color[3] = 1.f;
		p.colorOpacity = s.colorOpacity;
		p.blurAmount = s.blurAmount;
		p.blurQuality = s.blurQuality;
		p.dimBrightness = s.dimBrightness;
		p.dimOpacity = s.dimOpacity;
		p.featherPx = s.featherPixels(h);
		p.decontaminate = s.decontaminate;
		p.haloRemoval = s.haloRemoval;
		p.upsampleQuality = s.upsampleQuality;
		p.rangeSigma = 0.08f + 0.10f * s.smoothness;
		p.debugView = s.debugView;
		uint32_t bgW = 0, bgH = 0;
		gs_texture_t *bg = nullptr;
		if (s.mode == BackgroundMode::Image || s.mode == BackgroundMode::Video || s.mode == BackgroundMode::Source)
			bg = ctx->background.texture(bgW, bgH);
		ctx->gpu.render(p, bg, bgW, bgH, !firstRenderOfFrame);
	}

	// 5. Developer overlay.
	if (s.overlay || developerModeEnv()) {
		updateOverlay(ctx, w, h);
		ctx->overlay.render(w, h);
	}
	if (firstRenderOfFrame)
		ctx->worker->performance().recordRender(sw.elapsedMs());
}

void filterRender(void *data, gs_effect_t *)
{
	static std::atomic<uint64_t> lastLogNs{0};
	guarded("filter_render", [&] { filterRenderImpl(data); }, &lastLogNs);
}

uint32_t filterWidth(void *data)
{
	auto *ctx = static_cast<FilterContext *>(data);
	obs_source_t *target = obs_filter_get_target(ctx->source);
	return target ? obs_source_get_base_width(target) : 0;
}

uint32_t filterHeight(void *data)
{
	auto *ctx = static_cast<FilterContext *>(data);
	obs_source_t *target = obs_filter_get_target(ctx->source);
	return target ? obs_source_get_base_height(target) : 0;
}

void filterHide(void *data)
{
	auto *ctx = static_cast<FilterContext *>(data);
	if (ctx->worker) {
		ctx->workerEnabled = false;
		ctx->worker->setEnabled(false);
	}
}

void filterShow(void *data)
{
	auto *ctx = static_cast<FilterContext *>(data);
	if (ctx->worker) {
		ctx->workerEnabled = true;
		ctx->worker->setEnabled(true);
		ctx->worker->resetTemporal();
	}
}

} // namespace

void registerFilter()
{
	static obs_source_info info = {};
	info.id = "promatte_filter";
	info.type = OBS_SOURCE_TYPE_FILTER;
	info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB;
	info.get_name = filterName;
	info.create = filterCreate;
	info.destroy = filterDestroy;
	info.update = filterUpdate;
	info.get_defaults = filterDefaults;
	info.get_properties = filterProperties;
	info.video_tick = filterTick;
	info.video_render = filterRender;
	info.get_width = filterWidth;
	info.get_height = filterHeight;
	info.hide = filterHide;
	info.show = filterShow;
	info.icon_type = OBS_ICON_TYPE_CAMERA;
	obs_register_source(&info);
}

} // namespace promatte
