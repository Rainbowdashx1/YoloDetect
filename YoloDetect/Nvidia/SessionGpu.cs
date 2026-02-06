using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Buffers;

namespace YoloDetect.Nvidia
{
    public class SessionGpu : IDisposable
    {
        public InferenceSession session;
        public DenseTensor<float> _reusableTensor = new(new[] { 1, 3, 640, 640 });
        public DenseTensor<float> _reusableTensorBatch = new(new[] { 2, 3, 640, 640 });

        // Tensores de salida reutilizables con IO Binding
        private DenseTensor<float>? _reusableOutputSingle;
        private DenseTensor<float>? _reusableOutputBatch;

        // IO Binding para zero-copy
        private OrtIoBinding? _ioBindingSingle;
        private OrtIoBinding? _ioBindingBatch;

        // OrtValues reutilizables (vinculan memoria pinneada)
        private OrtValue? _inputOrtValueSingle;
        private OrtValue? _inputOrtValueBatch;
        private OrtValue? _outputOrtValueSingle;
        private OrtValue? _outputOrtValueBatch;

        // Handles de memoria pinneada
        private MemoryHandle _inputHandleSingle;
        private MemoryHandle _inputHandleBatch;
        private MemoryHandle _outputHandleSingle;
        private MemoryHandle _outputHandleBatch;

        private RunOptions _runOptions;
        private bool _disposed;

        // Dimensiones de salida del modelo (se detectan en la primera ejecución)
        private int[]? _outputDimsSingle;
        private int[]? _outputDimsBatch;

        public SessionGpu(string modelPath) 
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL; // Paralelización completa
            sessionOptions.EnableMemoryPattern = true; // Optimización de patrones de memoria
            sessionOptions.EnableCpuMemArena = false; // Desactivar para GPU pura
            sessionOptions.EnableProfiling = false; // Sin profiling para máximo rendimiento

            sessionOptions.AddSessionConfigEntry("session.dynamic_block_base", "8"); // Bloques más grandes
            sessionOptions.AddSessionConfigEntry("session.use_env_allocators", "1"); // Allocators optimizados
            sessionOptions.AddSessionConfigEntry("session.disable_prepacking", "0"); // Habilitar prepacking

            sessionOptions.AddSessionConfigEntry("ep.cuda.device_id", "0"); // GPU principal
            sessionOptions.AddSessionConfigEntry("ep.cuda.arena_extend_strategy", "kSameAsRequested"); // Estrategia de memoria agresiva
            sessionOptions.AddSessionConfigEntry("ep.cuda.gpu_mem_limit", "0"); // Sin límite de memoria GPU
            sessionOptions.AddSessionConfigEntry("ep.cuda.cudnn_conv_algo_search", "EXHAUSTIVE"); // Búsqueda exhaustiva del mejor algoritmo

            sessionOptions.AddSessionConfigEntry("ep.cuda.do_copy_in_default_stream", "1"); // Copia en stream por defecto
            sessionOptions.AddSessionConfigEntry("ep.cuda.cudnn_conv1d_pad_to_nc1d", "1"); // Optimización de padding
            sessionOptions.AddSessionConfigEntry("ep.cuda.enable_cuda_graph", "1"); // CUDA Graphs para máximo rendimiento
            sessionOptions.AddSessionConfigEntry("ep.cuda.cudnn_conv_use_max_workspace", "1"); // Usar máximo workspace de cuDNN

            sessionOptions.AddSessionConfigEntry("ep.cuda.gpu_external_alloc", "0"); // Allocator interno para mejor rendimiento
            sessionOptions.AddSessionConfigEntry("ep.cuda.gpu_external_free", "0");
            sessionOptions.AddSessionConfigEntry("ep.cuda.gpu_external_empty_cache", "0");

            sessionOptions.AddSessionConfigEntry("ep.cuda.tunable_op_enable", "1"); // Habilitar ops tunables
            sessionOptions.AddSessionConfigEntry("ep.cuda.tunable_op_tuning_enable", "1"); // Auto-tuning activado
            sessionOptions.AddSessionConfigEntry("ep.cuda.user_compute_stream", "1"); // Stream de computación dedicado

            sessionOptions.AddSessionConfigEntry("session.set_denormal_as_zero", "1"); // Tratar denormales como cero
            sessionOptions.AddSessionConfigEntry("session.use_device_allocator_for_initializers", "1"); // Allocator GPU para inicializadores
            sessionOptions.AddSessionConfigEntry("session.inter_op_num_threads", "0"); // Usar todos los threads disponibles
            sessionOptions.AddSessionConfigEntry("session.intra_op_num_threads", "0"); // Auto-detección

            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL; // Todas las optimizaciones
            sessionOptions.AddSessionConfigEntry("optimization.minimal_build_optimizations", ""); // Sin restricciones

            sessionOptions.AppendExecutionProvider_CUDA(0);
            session = new InferenceSession(modelPath, sessionOptions);

            // Inicializar IO bindings y RunOptions (los tensores de salida se inicializan en la primera ejecución)
            _ioBindingSingle = session.CreateIoBinding();
            _ioBindingBatch = session.CreateIoBinding();
            _runOptions = new RunOptions();
        }

        /// <summary>
        /// Inicializa el binding para ejecución single (batch=1)
        /// </summary>
        private void InitializeSingleBinding(int[] outputDims)
        {
            _outputDimsSingle = outputDims;
            _reusableOutputSingle = new DenseTensor<float>(outputDims);

            // Pin de memoria para entrada y salida
            _inputHandleSingle = _reusableTensor.Buffer.Pin();
            _outputHandleSingle = _reusableOutputSingle.Buffer.Pin();

            // Crear OrtValues desde memoria pinneada
            var inputShape = new long[] { 1, 3, 640, 640 };
            var inputArray = System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(_reusableTensor.Buffer, out var inputSegment) 
                ? inputSegment.Array! 
                : _reusableTensor.Buffer.ToArray();
            _inputOrtValueSingle = OrtValue.CreateTensorValueFromMemory(inputArray, inputShape);

            var outputShape = outputDims.Select(d => (long)d).ToArray();
            var outputArray = System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(_reusableOutputSingle.Buffer, out var outputSegment)
                ? outputSegment.Array!
                : _reusableOutputSingle.Buffer.ToArray();
            _outputOrtValueSingle = OrtValue.CreateTensorValueFromMemory(outputArray, outputShape);

            // Vincular al IoBinding
            _ioBindingSingle!.BindInput("images", _inputOrtValueSingle);
            _ioBindingSingle.BindOutput("output0", _outputOrtValueSingle);
        }

        /// <summary>
        /// Inicializa el binding para ejecución batch (batch=2)
        /// </summary>
        private void InitializeBatchBinding(int[] outputDims)
        {
            _outputDimsBatch = outputDims;
            _reusableOutputBatch = new DenseTensor<float>(outputDims);

            // Pin de memoria para entrada y salida
            _inputHandleBatch = _reusableTensorBatch.Buffer.Pin();
            _outputHandleBatch = _reusableOutputBatch.Buffer.Pin();

            // Crear OrtValues desde memoria pinneada
            var inputShape = new long[] { 2, 3, 640, 640 };
            var inputArray = System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(_reusableTensorBatch.Buffer, out var inputSegment)
                ? inputSegment.Array!
                : _reusableTensorBatch.Buffer.ToArray();
            _inputOrtValueBatch = OrtValue.CreateTensorValueFromMemory(inputArray, inputShape);

            var outputShape = outputDims.Select(d => (long)d).ToArray();
            var outputArray = System.Runtime.InteropServices.MemoryMarshal.TryGetArray<float>(_reusableOutputBatch.Buffer, out var outputSegment)
                ? outputSegment.Array!
                : _reusableOutputBatch.Buffer.ToArray();
            _outputOrtValueBatch = OrtValue.CreateTensorValueFromMemory(outputArray, outputShape);

            // Vincular al IoBinding
            _ioBindingBatch!.BindInput("images", _inputOrtValueBatch);
            _ioBindingBatch.BindOutput("output0", _outputOrtValueBatch);
        }

        public DenseTensor<float>? SessionRun(Mat matframeLetterbox) 
        {
            TensorConverterSingle.MatToTensorHybridNoParallel(matframeLetterbox, _reusableTensor);

            // Primera ejecución: detectar dimensiones de salida e inicializar bindings
            if (_outputDimsSingle == null)
            {
                var inputs = new List<NamedOnnxValue>(1)
                {
                    NamedOnnxValue.CreateFromTensor("images", _reusableTensor)
                };
                using var results = session.Run(inputs);
                var outputTensor = results[0].AsTensor<float>() as DenseTensor<float>;

                if (outputTensor == null)
                    return null;

                InitializeSingleBinding(outputTensor.Dimensions.ToArray());
                outputTensor.Buffer.Span.CopyTo(_reusableOutputSingle!.Buffer.Span);
                return _reusableOutputSingle;
            }

            // Re-vincular entrada para forzar lectura de datos actualizados desde CPU
            _ioBindingSingle!.ClearBoundInputs();
            _ioBindingSingle.BindInput("images", _inputOrtValueSingle);

            // Ejecuciones subsecuentes: usar IO Binding
            session.RunWithBinding(_runOptions, _ioBindingSingle);
            return _reusableOutputSingle;
        }

        public DenseTensor<float>? SessionRunBatch(Mat mat1, Mat mat2)
        {
            TensorConverterBatch.MatToTensorHybridBatch(mat1, mat2, _reusableTensorBatch);

            if (_outputDimsBatch == null)
            {
                var inputs = new List<NamedOnnxValue>(1)
                {
                    NamedOnnxValue.CreateFromTensor("images", _reusableTensorBatch)
                };
                using var results = session.Run(inputs);
                var outputTensor = results.First(r => r.Name == "output0").AsTensor<float>() as DenseTensor<float>;

                if (outputTensor == null)
                    return null;

                InitializeBatchBinding(outputTensor.Dimensions.ToArray());
                outputTensor.Buffer.Span.CopyTo(_reusableOutputBatch!.Buffer.Span);
                return _reusableOutputBatch;
            }

            // Re-vincular entrada para forzar lectura de datos actualizados desde CPU
            _ioBindingBatch!.ClearBoundInputs();
            _ioBindingBatch.BindInput("images", _inputOrtValueBatch);

            // Ejecuciones subsecuentes: usar IO Binding
            session.RunWithBinding(_runOptions, _ioBindingBatch);
            return _reusableOutputBatch;
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            // Dispose OrtValues primero
            _inputOrtValueSingle?.Dispose();
            _inputOrtValueBatch?.Dispose();
            _outputOrtValueSingle?.Dispose();
            _outputOrtValueBatch?.Dispose();

            // Dispose handles de memoria
            _inputHandleSingle.Dispose();
            _inputHandleBatch.Dispose();
            _outputHandleSingle.Dispose();
            _outputHandleBatch.Dispose();
            _outputHandleSingle.Dispose();
            _outputHandleBatch.Dispose();

            _ioBindingSingle?.Dispose();
            _ioBindingBatch?.Dispose();
            _runOptions?.Dispose();
            session?.Dispose();

            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
