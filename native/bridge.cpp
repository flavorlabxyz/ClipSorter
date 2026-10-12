// Recipe Clip Sorter: Linux-only adapter to the separately installed official SDK.
// This program opens source clips and emits JSON/PPM to stdout. It has no media
// output path argument and never calls metadata setters, sidecar saves, or trim.
#include "BlackmagicRawAPI.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(HRESULT hr, const char* operation) {
    if (hr != S_OK) { std::ostringstream out; out << operation << " failed (0x" << std::hex << hr << ")"; throw std::runtime_error(out.str()); }
}
static std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) { const char* h = "0123456789abcdef"; out << "\\u00" << h[c >> 4] << h[c & 15]; }
        else out << c;
    }
    out << '"'; return out.str();
}
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** address() { return &p; }
};

class Callback final : public IBlackmagicRawCallback {
    std::atomic<ULONG> refs{1};
    std::mutex mutex;
    std::condition_variable condition;
    bool done = false;
    std::string error;
public:
    BlackmagicRawResolutionScale scale = blackmagicRawResolutionScaleEighth;
    IBlackmagicRawClipProcessingAttributes* attributes = nullptr;
    uint32_t width = 0, height = 0;
    std::vector<unsigned char> rgb;
    HRESULT QueryInterface(REFIID iid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        const REFIID unknown = IID_IUnknown;
        if (std::memcmp(&iid, &IID_IBlackmagicRawCallback, sizeof(iid)) == 0 || std::memcmp(&iid, &unknown, sizeof(iid)) == 0) {
            *out = static_cast<IBlackmagicRawCallback*>(this); AddRef(); return S_OK;
        }
        return E_NOINTERFACE;
    }
    ULONG AddRef() override { return ++refs; }
    ULONG Release() override { ULONG count = --refs; if (!count) delete this; return count; }
    void finish(const std::string& message = "") {
        { std::lock_guard<std::mutex> guard(mutex); error = message; done = true; }
        condition.notify_one();
    }
    void reset() { std::lock_guard<std::mutex> guard(mutex); done = false; error.clear(); rgb.clear(); }
    void wait() {
        std::unique_lock<std::mutex> guard(mutex);
        if (!condition.wait_for(guard, std::chrono::seconds(120), [this]{ return done; })) {
            std::cerr << "SDK frame callback timed out\n"; std::_Exit(2);
        }
        if (!error.empty()) throw std::runtime_error(error);
    }
    void ReadComplete(IBlackmagicRawJob* job, HRESULT result, IBlackmagicRawFrame* frame) override {
        try {
            check(result, "Read frame");
            if (!frame) throw std::runtime_error("SDK returned no frame");
            check(frame->SetResolutionScale(scale), "Preview resolution scale");
            check(frame->SetResourceFormat(blackmagicRawResourceFormatRGBAU8), "RGBA format");
            IBlackmagicRawJob* processing = nullptr;
            check(frame->CreateJobDecodeAndProcessFrame(attributes, nullptr, &processing), "Create decode/process job");
            HRESULT hr = processing->Submit();
            if (hr != S_OK) { processing->Release(); check(hr, "Submit decode/process"); }
        } catch (const std::exception& exc) { finish(exc.what()); }
        job->Release();
    }
    void ProcessComplete(IBlackmagicRawJob* job, HRESULT result, IBlackmagicRawProcessedImage* image) override {
        try {
            check(result, "Process frame");
            if (!image) throw std::runtime_error("SDK returned no processed image");
            void* data = nullptr; uint32_t size = 0;
            BlackmagicRawResourceType type; BlackmagicRawResourceFormat format;
            check(image->GetWidth(&width), "Width"); check(image->GetHeight(&height), "Height");
            check(image->GetResource(&data), "Image buffer"); check(image->GetResourceSizeBytes(&size), "Buffer size");
            check(image->GetResourceType(&type), "Buffer type"); check(image->GetResourceFormat(&format), "Buffer format");
            if (type != blackmagicRawResourceTypeBufferCPU || format != blackmagicRawResourceFormatRGBAU8 || !data || size != uint64_t(width) * height * 4)
                throw std::runtime_error("Expected tightly packed RGBA8 CPU output");
            rgb.resize(size_t(width) * height * 3);
            auto src = static_cast<unsigned char*>(data);
            for (size_t i = 0; i < size_t(width) * height; ++i) std::memcpy(rgb.data() + i * 3, src + i * 4, 3);
            finish();
        } catch (const std::exception& exc) { finish(exc.what()); }
        job->Release();
    }
    void DecodeComplete(IBlackmagicRawJob*, HRESULT) override {}
    void TrimProgress(IBlackmagicRawJob*, float) override {}
    void TrimComplete(IBlackmagicRawJob*, HRESULT) override {}
    void SidecarMetadataParseWarning(IBlackmagicRawClip*, const char*, uint32_t, const char* info) override { std::cerr << "Sidecar warning: " << (info ? info : "") << '\n'; }
    void SidecarMetadataParseError(IBlackmagicRawClip*, const char*, uint32_t, const char* info) override { std::cerr << "Sidecar error: " << (info ? info : "") << '\n'; }
    void PreparePipelineComplete(void*, HRESULT result) override { if (result != S_OK) finish("Pipeline preparation failed"); }
#ifdef RCS_BRAW_READ_AUDIO
    void ReadAudioComplete(IBlackmagicRawJob*, HRESULT, IBlackmagicRawAudioBuffer*) override {}
#endif
};

static std::string scalar(const Variant& value) {
    switch (value.vt) {
        case blackmagicRawVariantTypeString: return quote(value.bstrVal ? value.bstrVal : "");
        case blackmagicRawVariantTypeS16: return std::to_string(value.iVal);
        case blackmagicRawVariantTypeU16: return std::to_string(value.uiVal);
        case blackmagicRawVariantTypeS32: return std::to_string(value.intVal);
        case blackmagicRawVariantTypeU32: return std::to_string(value.uintVal);
        case blackmagicRawVariantTypeFloat32: return std::isfinite(value.fltVal) ? std::to_string(value.fltVal) : "null";
        default: return "null"; // Arrays/SDK-specific scalar types are not needed for V1 grouping.
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: recipe-braw diagnostic | probe SOURCE | frame SOURCE SECONDS SIZE | stream SOURCE SIZE FPS");
        const char* library = std::getenv("BLACKMAGIC_RAW_LIBRARY_DIR");
        Ref<IBlackmagicRawFactory> factory;
        factory.p = library ? CreateBlackmagicRawFactoryInstanceFromPath(library) : CreateBlackmagicRawFactoryInstance();
        if (!factory.p) { std::cerr << "Official Blackmagic RAW SDK could not load. Set BLACKMAGIC_RAW_LIBRARY_DIR to the SDK Linux/Libraries directory containing libBlackmagicRawAPI.so and its companion libraries.\n"; return 78; }
        Ref<IBlackmagicRaw> codec; check(factory->CreateCodec(codec.address()), "Create codec");
        Ref<IBlackmagicRawConfiguration> config;
        check(codec->QueryInterface(IID_IBlackmagicRawConfiguration, reinterpret_cast<void**>(config.address())), "Configuration");
        check(config->SetPipeline(blackmagicRawPipelineCPU, nullptr, nullptr), "CPU pipeline");
        check(config->SetCPUThreads(2), "CPU threads");
        if (std::string(argv[1]) == "diagnostic") {
            const char* version = nullptr; check(config->GetVersion(&version), "SDK version");
            std::cout << "{\"sdk_version\":" << quote(version ? version : "unknown") << ",\"pipeline\":\"CPU\",\"bridge_protocol\":1}\n"; return 0;
        }
        if (argc < 3) throw std::runtime_error("Source path is required");
        Ref<IBlackmagicRawClip> clip; check(codec->OpenClip(argv[2], clip.address()), "Open clip (read only)");
        uint32_t width = 0, height = 0; uint64_t count = 0; float fps = 0;
        check(clip->GetWidth(&width), "Clip width"); check(clip->GetHeight(&height), "Clip height");
        check(clip->GetFrameRate(&fps), "Clip frame rate"); check(clip->GetFrameCount(&count), "Clip frame count");
        if (!width || !height || !count || fps <= 0 || !std::isfinite(fps)) throw std::runtime_error("Invalid clip properties");
        if (std::string(argv[1]) == "probe") {
            const char* camera = nullptr; clip->GetCameraType(&camera);
            const char* timecode = nullptr; clip->GetTimecodeForFrame(0, &timecode);
            std::string recorded = "null";
            std::ostringstream metadata;
            metadata << "{\"frame_count\":" << count << ",\"timecode\":" << quote(timecode ? timecode : "");
            Ref<IBlackmagicRawMetadataIterator> iterator;
            if (clip->GetMetadataIterator(iterator.address()) == S_OK) {
                do {
                    const char* key = nullptr; Variant value{}; VariantInit(&value);
                    if (iterator->GetKey(&key) == S_OK && key && iterator->GetData(&value) == S_OK) {
                        std::string field(key), encoded = scalar(value);
                        metadata << ',' << quote(field) << ':' << encoded;
                        if (field == "recordingDateTime" || field == "creation_time") recorded = encoded;
                    }
                    VariantClear(&value);
                } while (iterator->Next() == S_OK);
            }
            metadata << '}';
            std::cout << "{\"duration\":" << double(count) / fps << ",\"fps\":" << fps << ",\"width\":" << width << ",\"height\":" << height
                      << ",\"codec\":\"Blackmagic RAW\",\"camera\":" << (camera ? quote(camera) : "null") << ",\"recorded_at\":" << recorded << ",\"extra\":" << metadata.str() << "}\n";
            return 0;
        }
        if (argc != 5) throw std::runtime_error("frame/stream requires two numeric arguments");
        bool stream = std::string(argv[1]) == "stream";
        if (!stream && std::string(argv[1]) != "frame") throw std::runtime_error("Unknown command");
        int size = std::stoi(argv[stream ? 3 : 4]);
        if (size < 64 || size > 2048) throw std::runtime_error("Preview size must be 64–2048");
        Ref<IBlackmagicRawClipProcessingAttributes> attributes;
        check(clip->CloneClipProcessingAttributes(attributes.address()), "Clone display attributes");
        // Processing overrides affect this decode only; no clip metadata is changed.
        for (auto attribute : {blackmagicRawClipProcessingAttributeGamma, blackmagicRawClipProcessingAttributeGamut}) {
            Variant value{}; value.vt = blackmagicRawVariantTypeString; value.bstrVal = "Rec.709";
            if (attributes->SetClipAttribute(attribute, &value) != S_OK) std::cerr << "Rec.709 override unavailable; SDK clip default retained\n";
        }
        Ref<Callback> callback; callback.p = new Callback(); callback->attributes = attributes.p;
        auto longest = std::max(width, height);
        callback->scale = longest / 8 >= uint32_t(size) ? blackmagicRawResolutionScaleEighth : longest / 4 >= uint32_t(size) ? blackmagicRawResolutionScaleQuarter : longest / 2 >= uint32_t(size) ? blackmagicRawResolutionScaleHalf : blackmagicRawResolutionScaleFull;
        check(codec->SetCallback(callback.p), "Set callback");
        const double out_fps = stream ? std::stod(argv[4]) : 1;
        const double start = stream ? 0 : std::stod(argv[3]);
        if (!std::isfinite(start) || !std::isfinite(out_fps) || start < 0 || out_fps <= 0 || out_fps > 30) throw std::runtime_error("Invalid sampling position/rate");
        const uint64_t samples = stream ? uint64_t(std::ceil(double(count) / fps * out_fps)) : 1;
        for (uint64_t i = 0; i < samples; ++i) {
            uint64_t index = std::min(count - 1, uint64_t((start + i / out_fps) * fps));
            callback->reset(); IBlackmagicRawJob* job = nullptr;
            check(clip->CreateJobReadFrame(index, &job), "Create read job");
            HRESULT hr = job->Submit(); if (hr != S_OK) { job->Release(); check(hr, "Submit read"); }
            callback->wait(); check(codec->FlushJobs(), "Flush jobs");
            // SDK scaling is performed before processing. PPM contains reduced RGB.
            std::cout << "P6\n" << callback->width << ' ' << callback->height << "\n255\n";
            std::cout.write(reinterpret_cast<const char*>(callback->rgb.data()), callback->rgb.size());
            if (!std::cout) throw std::runtime_error("Preview output pipe closed");
        }
        check(codec->SetCallback(nullptr), "Clear callback");
        return 0;
    } catch (const std::exception& exc) { std::cerr << exc.what() << '\n'; return 2; }
}
