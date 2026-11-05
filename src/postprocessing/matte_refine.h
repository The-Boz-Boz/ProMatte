#pragma once
#include <cstdint>
#include <vector>

namespace promatte {

struct RefineParams {
	// Foreground/background separation: alpha values are re-mapped through a soft
	// threshold centred on `threshold` with half-width `softness`. Values well below
	// become 0, well above become 1, the band in between keeps a smooth ramp (hair).
	float threshold = 0.5f; // 0.05..0.95
	float softness = 0.2f;  // 0..0.5 (0 = hard cut)
	// Morphological shift in AI-resolution pixels: negative erodes (removes halo),
	// positive dilates (protects thin structures).
	int shiftPixels = 0; // -4..4
	// Remove disconnected foreground blobs smaller than this fraction of the frame.
	float minComponentArea = 0.0f; // 0 = disabled, e.g. 0.01
	// Fill enclosed background holes smaller than this fraction of the frame.
	float maxHoleArea = 0.0f; // 0 = disabled

	// Alpha-matting model (ModelDescriptor::isAlphaMatting, set by the worker):
	// instead of the threshold curve only the tails are cut - alpha < matteLo
	// becomes 0, alpha > matteHi becomes 1, the range in between is stretched
	// linearly, so semi-transparent hair keeps its alpha. threshold/softness are
	// ignored. Enclosed holes the model itself rates as background (mean alpha
	// below kMattingHoleKeepAlpha) are not filled: they are real gaps (between
	// the fingers, between an arm and the body), not misses.
	bool alphaMatting = false;
	float matteLo = 0.04f; // 0.02..0.15, from the separation slider
	float matteHi = 0.96f;
};

constexpr float kMattingHoleKeepAlpha = 0.2f;

// CPU refinement of the matte at AI resolution (cheap: ~150-500k pixels).
// Full-resolution, edge-aware refinement happens on the GPU afterwards.
class MatteRefiner {
public:
	void process(float *alpha, uint32_t w, uint32_t h, const RefineParams &p);

	// Individual stages (public for unit tests).
	static void applyConfidenceCurve(float *alpha, size_t n, float threshold, float softness);
	// Tail cut for alpha-matting models: [lo, hi] -> [0, 1] linearly, clamped.
	static void applyMatteTails(float *alpha, size_t n, float lo, float hi);
	void morphShift(float *alpha, uint32_t w, uint32_t h, int shiftPixels);
	// Connected-component filtering on the binarised matte (alpha > 0.5).
	// Holes whose mean alpha is below `holeKeepAlpha` are left alone (0 = fill
	// every small hole). Returns the number of removed foreground components.
	int filterComponents(float *alpha, uint32_t w, uint32_t h, float minComponentArea, float maxHoleArea,
			     float holeKeepAlpha = 0.f);

private:
	std::vector<float> tmp_;
	std::vector<int32_t> labels_;
	std::vector<uint32_t> stack_;
};

} // namespace promatte
