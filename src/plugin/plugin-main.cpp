// ProMatte - Professional Real-Time AI Background Removal for OBS Studio
// Module entry point.
#include <cstdlib>
#include <memory>
#include <string>

#include <obs-module.h>
#include <util/platform.h>

#include "inference/backend_registry.h"
#include "models/model_manager.h"
#include "performance/capability_store.h"
#include "obs/filter.h"
#include "utils/logging.h"
#include "utils/system_info.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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

#ifdef _WIN32
// ONNX Runtime delay-loads DirectML.dll by bare name. Once the plugin is loaded
// its directory is no longer on the search path, so Windows would hand out
// System32\DirectML.dll (the OS copy) or whatever another plugin loaded first.
// Load the copy that ships next to promatte.dll up front so the bare-name lookup
// resolves to it, and log which file is actually in use.
void preloadDirectML()
{
	HMODULE self = nullptr;
	if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(&preloadDirectML), &self))
		return;
	wchar_t buf[2048];
	DWORD n = GetModuleFileNameW(self, buf, 2048);
	if (n == 0 || n >= 2048)
		return;
	std::wstring path(buf, n);
	path = path.substr(0, path.find_last_of(L"\\/") + 1) + L"DirectML.dll";
	HMODULE dml = GetModuleHandleW(L"DirectML.dll");
	const bool preexisting = dml != nullptr;
	if (!dml)
		dml = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
	if (!dml) {
		PM_LOG_WARN("could not load bundled DirectML.dll (error %lu)", GetLastError());
		return;
	}
	n = GetModuleFileNameW(dml, buf, 2048);
	char utf8[4096] = {};
	if (n > 0 && n < 2048)
		WideCharToMultiByte(CP_UTF8, 0, buf, (int)n, utf8, sizeof(utf8) - 1, nullptr, nullptr);
	PM_LOG_INFO("DirectML.dll in use: %s%s", utf8, preexisting ? " (already loaded by another module)" : "");
}
#endif
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
#ifdef _WIN32
	preloadDirectML();
#endif
	PM_LOG_INFO("system: %s, %u threads, %.1f GB RAM, %s", sysinfo::cpuName().c_str(), sysinfo::logicalCores(),
		    double(sysinfo::totalPhysicalMemory()) / (1024.0 * 1024 * 1024), sysinfo::osVersion().c_str());
	for (const auto &g : sysinfo::enumerateGpus())
		PM_LOG_INFO("gpu %d: %s (%s, %.0f MB dedicated)%s", g.adapterIndex, g.name.c_str(),
			    sysinfo::vendorName(g.vendorId), double(g.dedicatedVideoMemory) / (1024.0 * 1024),
			    g.isSoftware ? " [software]" : "");

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
