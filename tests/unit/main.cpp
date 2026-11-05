#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#include <cstdio>

#include <onnxruntime_cxx_api.h>

#include "utils/logging.h"

int main(int argc, char **argv)
{
	promatte::log::setMinLevel(promatte::log::Level::Warning);
	// promatte-core is built with ORT_API_MANUAL_INIT (see CMakeLists.txt).
	const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (!api) {
		std::fprintf(stderr, "ONNX Runtime %s does not provide API version %d\n",
			     OrtGetApiBase()->GetVersionString(), ORT_API_VERSION);
		return 1;
	}
	Ort::InitApi(api);
	doctest::Context ctx;
	ctx.applyCommandLine(argc, argv);
	return ctx.run();
}
