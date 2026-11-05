#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "inference/segmentation_backend.h"

namespace promatte {

// ONNX Runtime backend. All GPU vendors are reached through execution providers:
//   DirectML (Windows, any D3D12 GPU), CUDA / TensorRT (when an ORT build with those
//   providers is installed), CoreML (macOS), OpenVINO (Intel) and CPU.
// Which providers exist is discovered at runtime; nothing is assumed.
class OnnxRuntimeBackend final : public SegmentationBackend {
public:
	explicit OnnxRuntimeBackend(BackendKind kind);
	~OnnxRuntimeBackend() override;

	bool initialize(const BackendConfig &config, const ModelDescriptor &model, const std::string &modelPath,
			std::string &error) override;
	bool processFrame(const InferenceInput &input, InferenceOutput &output, std::string &error) override;
	void shutdown() override;
	BackendCapabilities capabilities() const override { return caps_; }
	void resetState() override;
	void cancel() override;
	bool isInitialized() const override { return session_ != nullptr; }

	// Shared process-wide environment (thread-safe, created on first use).
	static Ort::Env &env();
	static std::vector<std::string> availableProviders();
	// Windows DirectML builds: makes sure a DirectML.dll is loaded before ONNX
	// Runtime needs it - the copy already in the process if there is one,
	// otherwise the one next to this module (never a System32 search). Logs its
	// path and FileVersion once. Returns false when DirectML must not be used
	// (missing, or older than 1.15). Always true in other builds.
	static bool prepareDirectML();

private:
	struct IoNames {
		std::string src;
		std::vector<std::string> recIn;  // r1i..r4i
		std::vector<std::string> recOut; // r1o..r4o
		std::string ratio;               // downsample_ratio
		std::string alpha;               // pha / output
		std::string foreground;          // fgr (optional)
		// Element types from the graph. src and the recurrent state share one type
		// (FLOAT or FLOAT16); downsample_ratio is always FLOAT.
		ONNXTensorElementDataType srcType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		ONNXTensorElementDataType alphaType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		ONNXTensorElementDataType foregroundType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	};

	bool discoverIo(std::string &error);
	bool runRecurrent(const InferenceInput &input, InferenceOutput &output, std::string &error);
	bool runSimple(const InferenceInput &input, InferenceOutput &output, std::string &error);
	void resetRecurrentState();
	// Wraps the preprocessed frame as the model's src tensor (float or float16).
	Ort::Value makeSrcTensor(const InferenceInput &input);
	// Float view of a FLOAT or FLOAT16 output tensor (converted into `scratch`).
	const float *floatData(const Ort::Value &v, size_t count, std::vector<float> &scratch, std::string &error) const;

	BackendKind kind_;
	BackendCapabilities caps_;
	BackendConfig config_;
	ModelDescriptor model_;
	std::unique_ptr<Ort::Session> session_;
	std::unique_ptr<Ort::SessionOptions> sessionOptions_;
	Ort::RunOptions runOptions_;
	Ort::MemoryInfo cpuMemory_;
	std::unique_ptr<Ort::MemoryInfo> deviceMemory_;
	std::unique_ptr<Ort::IoBinding> binding_;
	IoNames io_;
	std::vector<Ort::Value> recState_; // recurrent state values (device or CPU)
	std::vector<float> zeroState_;
	uint16_t zeroStateHalf_ = 0; // float16 +0.0
	std::vector<uint16_t> halfInput_; // float -> float16 when a caller only provided float data
	std::vector<float> alphaScratch_, foregroundScratch_;
	bool useDeviceBinding_ = false;
	bool stateValid_ = false;
	uint32_t stateW_ = 0, stateH_ = 0;
	float stateRatio_ = 0.f;
	std::atomic<bool> cancelled_{false};

	// D3D12/DirectML objects created for explicit adapter selection (DML1 API).
	// Only defined in the Windows DirectML build; a unique_ptr to a type that is
	// never completed will not compile its deleter elsewhere.
#if defined(_WIN32) && defined(PROMATTE_HAVE_DIRECTML)
	struct DmlDevice;
	std::unique_ptr<DmlDevice> dmlDevice_;
	bool appendDirectMLProvider(Ort::SessionOptions &so, int adapterIndex, std::string &deviceName);
#endif
};

} // namespace promatte
