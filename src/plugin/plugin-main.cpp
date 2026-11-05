// ProMatte - Professional Real-Time AI Background Removal for OBS Studio
// Module entry point.
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <obs-module.h>
#include <util/platform.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "backends/onnxruntime_backend.h"
#include "inference/backend_registry.h"
#include "models/model_manager.h"
#include "performance/capability_store.h"
#include "obs/filter.h"
#include "utils/file_utils.h"
#include "utils/logging.h"
#include "utils/system_info.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("promatte", "en-US")

namespace promatte {

namespace {
std::unique_ptr<ModelManager> g_models;
std::unique_ptr<CapabilityStore> g_capabilities;
bool g_devMode = false;

void obsLogSink(log::Level level, const char *message)
{
	int lvl = level == log::Level::Error ? LOG_ERROR : level == log::Level::Warning ? LOG_WARNING : level == log::Level::Info ? LOG_INFO : LOG_DEBUG;
	blog(lvl, "[promatte] %s", message);
}

// Full path of the onnxruntime.dll this process actually uses (another plugin may
// have loaded a different copy before us).
std::string loadedOrtPath()
{
#ifdef _WIN32
	HMODULE ort = GetModuleHandleW(L"onnxruntime.dll");
	if (!ort)
		return "onnxruntime.dll (not found among loaded modules)";
	std::vector<wchar_t> buf(32768);
	const DWORD n = GetModuleFileNameW(ort, buf.data(), DWORD(buf.size()));
	if (n == 0 || n >= buf.size())
		return "onnxruntime.dll (path unavailable)";
	return fs::fromWide(std::wstring(buf.data(), n));
#else
	return "the shared onnxruntime library";
#endif
}
} // namespace

ModelManager &modelManager()
{
	return *g_models;
}

CapabilityStore &capabilityStore()
{
	return *g_capabilities;
}

bool developerModeEnv()
{
	return g_devMode;
}

const char *moduleDataPath(const char *file, std::string &storage)
{
	char *p = obs_module_file(file);
	if (!p)
		return nullptr;
	storage = p;
	bfree(p);
	return storage.c_str();
}

} // namespace promatte

MODULE_EXPORT const char *obs_module_name(void)
{
	return "ProMatte AI Background Removal";
}

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Professional real-time AI background removal, blur and replacement for webcams. Fully local; no cloud.";
}

bool obs_module_load(void)
{
	using namespace promatte;
	log::setSink(&obsLogSink);
	const char *env = std::getenv("PROMATTE_DEBUG");
	g_devMode = env && *env && std::string(env) != "0";
	log::setMinLevel(g_devMode ? log::Level::Debug : log::Level::Info);

	PM_LOG_INFO("ProMatte %s loading (OBS %s)", PROMATTE_VERSION, obs_get_version_string());
	PM_LOG_INFO("system: %s, %u threads, %.1f GB RAM, %s", sysinfo::cpuName().c_str(), sysinfo::logicalCores(),
		    double(sysinfo::totalPhysicalMemory()) / (1024.0 * 1024 * 1024), sysinfo::osVersion().c_str());
	for (const auto &g : sysinfo::enumerateGpus())
		PM_LOG_INFO("gpu %d: %s (%s, %.0f MB dedicated)%s", g.adapterIndex, g.name.c_str(),
			    sysinfo::vendorName(g.vendorId), double(g.dedicatedVideoMemory) / (1024.0 * 1024),
			    g.isSoftware ? " [software]" : "");

	// The core is built with ORT_API_MANUAL_INIT, so nothing has touched ONNX
	// Runtime yet. If an older onnxruntime.dll was loaded into OBS first (another
	// plugin), it cannot serve our API version: GetApi returns null and the first
	// Ort:: call would crash OBS. Stay loaded but inactive and say why.
	const OrtApiBase *ortBase = OrtGetApiBase();
	const OrtApi *ortApi = ortBase ? ortBase->GetApi(ORT_API_VERSION) : nullptr;
	const std::string ortPath = loadedOrtPath();
	if (!ortApi) {
		PM_LOG_ERROR("ONNX Runtime %s loaded from %s does not provide API version %d, which ProMatte needs. "
			     "Another plugin most likely loaded an older onnxruntime.dll first. The ProMatte filter is "
			     "NOT available in this session.",
			     ortBase ? ortBase->GetVersionString() : "(unknown version)", ortPath.c_str(), ORT_API_VERSION);
		return true;
	}
	Ort::InitApi(ortApi);
	PM_LOG_INFO("ONNX Runtime %s (API %d) loaded from %s", ortBase->GetVersionString(), ORT_API_VERSION,
		    ortPath.c_str());

	std::string bundled, user;
	moduleDataPath("models", bundled);
	char *cfg = obs_module_config_path("models");
	if (cfg) {
		user = cfg;
		bfree(cfg);
	}
	g_models = std::make_unique<ModelManager>(bundled, user);
	std::string manifest, err;
	if (moduleDataPath("models/manifest.json", manifest) && g_models->loadManifestFile(manifest, err)) {
		int installed = 0;
		for (const auto &m : g_models->list())
			installed += m.installed ? 1 : 0;
		PM_LOG_INFO("model manifest loaded: %zu approved models, %d installed (user dir: %s)",
			    g_models->list().size(), installed, user.c_str());
	} else {
		PM_LOG_ERROR("model manifest unavailable: %s", err.c_str());
	}

	std::string capPath;
	if (char *cp = obs_module_config_path("capability.json")) {
		capPath = cp;
		bfree(cp);
	}
	g_capabilities = std::make_unique<CapabilityStore>(capPath);
	if (g_capabilities->load())
		PM_LOG_INFO("capability store loaded: %zu measurements (%s)", g_capabilities->size(), capPath.c_str());

	// A second copy of the plugin (for example one left in
	// %ProgramData%\obs-studio\plugins after installing into the OBS folder)
	// loads fine but its obs_register_source call is rejected, so OBS silently
	// keeps whichever copy loaded first. That makes an update look like it did
	// nothing. Detect it and say so loudly, naming both files.
	if (obs_source_get_display_name("promatte_filter")) {
		const char *self = obs_get_module_binary_path(obs_current_module());
		PM_LOG_ERROR("ANOTHER COPY OF PROMATTE IS ALREADY LOADED. This copy (%s) will be IGNORED by OBS, "
			     "which keeps the copy that loaded first. Uninstall or delete one of them - a plugin "
			     "must exist either in the OBS folder (obs-plugins\\64bit) or in "
			     "%%ProgramData%%\\obs-studio\\plugins\\promatte, never both.",
			     self ? self : "unknown path");
		PM_LOG_INFO("ProMatte %s loaded but inactive (duplicate install)", PROMATTE_VERSION);
		return true;
	}

	// Load and check DirectML.dll before anything creates a DML session, so
	// onnxruntime.dll binds to this copy rather than whatever the search path finds.
	OnnxRuntimeBackend::prepareDirectML();
	// Probe backends once at load so the first filter starts quickly.
	enumerateBackends();
	registerFilter();
	PM_LOG_INFO("ProMatte loaded. Privacy: all processing is local; no frames leave this machine.");
	return true;
}

void obs_module_unload(void)
{
	if (promatte::g_capabilities)
		promatte::g_capabilities->save();
	promatte::g_capabilities.reset();
	promatte::g_models.reset();
	PM_LOG_INFO("ProMatte unloaded");
}
