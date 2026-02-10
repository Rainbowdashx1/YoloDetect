# YOLODetect (YOLOPerson)

## Description

**YoloDetect** is a high-performance C#/.NET 8 toolkit built around ONNX Runtime (GPU) and OpenCvSharp, designed to run custom YOLO models with an optimized end-to-end pipeline. Instead of being limited to person detection (the original YOLOPerson scope), YoloDetect aims to be a reusable detection framework where anyone can plug in their trained model and benefit from best-practice preprocessing and postprocessing techniques to maximize both speed and output quality.

## System Used

### Hardware
- **Processor**: Intel Core i7-8700 CPU @ 3.20GHz (Coffee Lake)
  - 6 physical cores, 12 logical cores
- **Operating System**: Windows 10 (22H2/2022 Update)
- **GPU**: NVIDIA GTX1070 8GB

### Software
- **.NET Version**: .NET 8.0.7
- **Microsoft.ML.OnnxRuntime.Gpu**: v1.18.1
- **OpenCvSharp4**
- **CUDA Toolkit**: 11.8 ([download](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe))
- **cuDNN**: 8.9.7 for CUDA 11.x ([download](https://developer.nvidia.com/cudnn-downloads))

## Setup and Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Rainbowdashx1/YoloDetect.git
   cd YoloDetect
   ```

2. **Install dependencies**
   Ensure you have installed:
   - .NET Core SDK 8.0.7
   - NVIDIA CUDA Toolkit 11.8
   - cuDNN 8.9.7 compatible with CUDA 11.x

3. **Build the project**

   ```bash
   dotnet build
   ```

4. **Run the project**

   ```bash
   dotnet run
   ```

## Future Improvements

- Add support for real-time detection from a camera.
- Optimize video processing performance.
- Integrate visualization directly into the project interface.
- Incorporate ByteTrack for advanced tracking capabilities.

---

# Changelog

## Version 1.0.11

### Target Class Filtering for YOLO Detection

Support has been added to filter detections by a set of target classes (`targetClasses`) in both **YOLOv11** and **YOLOv26** preprocessing pipelines.
Method and constructor signatures were updated to accept a `HashSet<int>` instead of a single class parameter. This allows dynamic filtering based on multiple specified classes, providing greater flexibility and customization during detection.
Interfaces and detection processors were updated accordingly to propagate the new parameter across the processing pipeline.

## Version 1.0.10

### Optimized Border Color Handling and Buffer Reuse in ProcessFrame

- Added a reusable _borderColor field to avoid repeated Scalar instantiations.
Improved _resizedBuffer reuse logic by validating width and height independently instead of relying on a single size comparison.

- These changes were applied to the LetterboxOptimized process.
While they do not result in a measurable speed improvement, they eliminate unnecessary object allocations on every frame, improving memory efficiency and reducing GC pressure.

## Version 1.0.9

### IO Binding Optimization
- **OrtIoBinding** to eliminate memory allocations during inference.

## Version 1.0.8

### Batch Post-Processing with Reusable Buffers
- Batch post-processing was refactored to operate directly on preallocated detection buffers instead of creating and returning new lists. Separate reusable buffers are maintained for left, right, and merged detections. Processing and merge steps now modify these buffers in place, and rendering uses the merged buffer as the single source of truth, reducing allocations and simplifying data flow

### Modular YOLO Detection Processor Architecture
- A processor-based architecture was introduced using the IDetectionProcessor interface to decouple post-processing logic from specific YOLO model versions. Dedicated processors for YOLOv11 and YOLOv26, along with a factory and model type enumeration, allow the capture pipeline to delegate post-processing cleanly, improving maintainability and scalability.

### Optimized Tensor Reuse and Memory Management
- ONNX inference was optimized by reusing DenseTensor<float> instances instead of allocating new tensors per inference. Post-processing now accesses tensor data through ReadOnlySpan<float> to work directly on the internal buffer, significantly reducing memory allocations and preventing potential memory leaks during GPU inference.

## Version 1.0.7

### Add two-batch yolov26
- The two-batch functionality has been added to yolov26

## Version 1.0.6

### Add MatToTensorHybridNoParallel
- A new method, MatToTensorHybridNoParallel, has been added.

### Add BenchMark
- BenchMark of the new method is added.

## Version 1.0.5

### Change namespace
- The namespaces of YoloPerson are changed to YoloDetect.

### Add BenchMark
- A new benchmark was added for the letterbox, which turned out to be no better than the current one.

## Version 1.0.4

### Project rename: YOLOPerson → YoloDetect
- Renamed the project to **YoloDetect** to better represent its purpose beyond person-only detection.
- Updated documentation to reflect the new identity and long-term direction.

### New project focus: plug-and-play custom models + best-practice pipeline
- Shifted the project vision toward a **reusable detection framework**:
  - Load and run **custom trained models** (YOLO/ONNX).
  - Apply **optimized preprocessing** (e.g., letterbox variants) for consistent input handling.
  - Apply **optimized postprocessing** for fast and accurate results.
- Goal: provide a baseline that is both **fast** (low allocations, efficient conversions, GPU-friendly) and **high quality** (stable preprocessing, solid postprocessing).

### Add Version
- The version was added to maintain a consistent order with the versions

## Version 1.0.3

### Letterbox preprocessing improved again
- Added a new method: **`LetterboxOptimized`**
- According to benchmarks, **`LetterboxOptimized` outperforms the previous implementation**
- Even though the difference can be in the **nanosecond range**, the optimized version is still **faster and more efficient**

### New and expanded benchmarks for Mat → Tensor
- Added **new benchmarks** focused on **Mat → Tensor conversion**
- Goal: **find the fastest and most optimal approach**
- Added **more conversion methods** and **new strategies** to speed up the conversion further

### Benchmark & conversion code refactor
- Benchmarks and Mat → Tensor conversion code were **separated and reorganized**
- The conversion logic is now split into specialized handlers:
  - **Single-image pipeline**
  - **Two-image pipeline**

### Reuse buffers to reduce allocations and CPU pressure
- Added multiple optimizations by **reusing buffers** (lists/tensors) instead of re-creating them on every inference
- Benefits:
  - Better runtime performance
  - Less CPU overhead from GC/allocation churn

### YOLOv26 support (Batch = 1)
- Added support for **YOLOv26 models with batch size = 1**
- A **batch size = 2 model was added**, but **support is not implemented yet** (pending)


## Version 1.0.2 - Two-Batch Processing 

#### Added
- **Multi-Model Support**
  - Added `yolo11m2batch.onnx` - YOLOv11 Medium with 2-batch processing (two-batch mode)
  - Added `yolo11n1batch.onnx` - YOLOv11 Nano optimized for single-batch processing
  - Added `yolo11n2batch.onnx` - YOLOv11 Nano with 2-batch processing (two-batch mode)
  
- **Interactive Model Selection in `Program.cs`**
  - Implemented a menu-driven interface allowing users to select between different YOLO models at runtime
  - Options include single-batch and dual-batch (two-batch) processing modes for both Medium and Nano variants

- **Batch Processing Methods in `Capture.cs`**
  - `runWithModel1Batch()` - Optimized pipeline for single-batch models
  - `runWithModel2Batch()` - Specialized pipeline for dual-batch models with overlapping region processing
  - `ProcessFrameBatchOverLap()` - Handles frame splitting, batch inference, and detection merging
  - `MergeOverlappingDetections()` - IoU-based duplicate detection elimination in overlapping regions

- **Batch Output Processing in `Preprocessed.cs`**
  - New method `PreproccessedOutputBatchOptimized()` for efficient handling of dual-batch inference results
  - Optimized memory layout for processing two simultaneous inference outputs
  - 
#### Changed
- **Refactored Video Capture Initialization in `Capture.cs`**
  - Extracted `VideoCapture()` method to eliminate code duplication
  - Now returns tuple `(VideoCapture, VideoWriter)` for reuse across different processing modes
  - Added proper resource disposal with `try-finally` blocks in both batch methods

- **Enhanced GPU Configuration in `SessionGpu.cs`**
  - Added aggressive CUDA optimization parameters for improved inference performance
  - Configured memory allocation strategies (`arena_extend_strategy`, `gpu_mem_limit`)
  - Enabled CUDA Graphs (`enable_cuda_graph`) for reduced kernel launch overhead
  - Optimized cuDNN convolution algorithm search (`cudnn_conv_algo_search: EXHAUSTIVE`)
  - Fine-tuned thread management (`InterOpNumThreads`, `IntraOpNumThreads`) for batch processing

#### Fixed
- **Corrected CUDA and ONNX Runtime Version Documentation**
  - Previous documentation listed incorrect versions for CUDA Toolkit and ONNX Runtime
  - Updated to reflect actual tested versions:
    - CUDA Toolkit: `11.x`
    - ONNX Runtime GPU: `1.18.1`

## Version 1.0.1

This version introduces performance optimizations, additional functionality, and improvements in code quality and readability.

### Added
- **`BenchMarksMethods` Project**  
  A new project was added to provide a benchmarking layer, enabling testing of individual methods to determine the most efficient in terms of execution time and CPU usage.
  
- **`MatToTensorParallel` Method in `SessionGpu`**  
  A new method for creating tensors more quickly, with improved execution time in milliseconds compared to the previous method.

### Changed
- **Enhanced `SessionGpu` Constructor**  
  Additional configuration parameters were introduced, resulting in a slight improvement in inference speed.

- **Updated Preprocessing in `PreProcessed.cs`**  
  Adjusted the preprocessing logic to focus exclusively on detecting people, as this is the only required functionality for this implementation. Previously, it iterated over all possible objects YOLOv11 could detect.

- **Comments Updated to English**  
  All comments were reviewed and converted to English for improved clarity and consistency.

### Removed
- **Unnecessary Comments**  
  Redundant or outdated comments were removed for better code readability.

---

## Version 1.0.0
- Initial release with foundational functionality and YOLOv11 integration.

