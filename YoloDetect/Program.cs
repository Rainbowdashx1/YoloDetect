using YoloDetect.VideoCapture;
using YoloDetect.VideoCapture.ProcesorDetection;
using YoloDetect.VideoSources;
using YoloDetect.Models;

internal class Program
{
    static private string yolo11m = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11m.onnx");
    static private string yolo11m2batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11m2batch.onnx");
    static private string yolo11n1batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n1batch.onnx");
    static private string yolo11n2batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo11n2batch.onnx");
    static private string yolo26n1batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo26n1batch.onnx");
    static private string yolo26n2batch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ModelOnnx", "yolo26n2batch.onnx");

    static bool batch = false;

    static private string modelPath = string.Empty;
    static private string videoPath = string.Empty;
    static private string? videoProcessPath;
    static private VideoSourceType? sourceType;
    private static ModelType modelType;
    private static void Main(string[] args)
    {
        Console.WriteLine("=== YoloPerson Detection ===\n");
        // Se descomprimira Ffmpeg
        FFmpegHelper.Initialize();
        
        if (FFmpegHelper.IsAvailable)
        {
            Console.WriteLine("FFmpeg disponible\n");
        }
        else
        {
            Console.WriteLine("FFmpeg NO disponible (las opciones 3 y 5 no funcionarán)\n");
        }

        Console.WriteLine("=== Selecciona fuente de video ===");
        Console.WriteLine("1. Archivo local (people-walking.mp4)");
        Console.WriteLine($"2. Stream RTSP (con OpenCvSharp)");
        Console.WriteLine($"3. Stream RTSP (con FFmpeg - baja latencia) {(FFmpegHelper.IsAvailable ? "" : "[NO DISPONIBLE]")}");
        Console.WriteLine($"4. Stream MJPEG (con OpenCvSharp)");
        Console.WriteLine($"5. Stream MJPEG (con FFmpeg - baja latencia) {(FFmpegHelper.IsAvailable ? "" : "[NO DISPONIBLE]")}");
        Console.Write("\nOpción: ");

        string? sourceOption = Console.ReadLine();

        switch (sourceOption)
        {
            case "1":
                videoPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Video", "people-walking.mp4");
                videoProcessPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Video", "people-walking_Processv3.mp4");
                sourceType = VideoSourceType.File;
                break;
            case "2":
                Console.Write("Ingresa URL RTSP (ej: rtsp://192.168.1.100:554/stream): ");
                videoPath = Console.ReadLine() ?? "";
                videoProcessPath = null;
                sourceType = VideoSourceType.RtspOpenCV;
                break;
            case "3":
                if (!FFmpegHelper.IsAvailable)
                {
                    Console.WriteLine("\nFFmpeg no está disponible. Usa la opción 2 en su lugar.");
                    return;
                }
                Console.Write("Ingresa URL RTSP (ej: rtsp://192.168.1.100:554/stream): ");
                videoPath = Console.ReadLine() ?? "";
                videoProcessPath = null;
                sourceType = VideoSourceType.RtspFFmpeg;
                break;
            case "4":
                Console.Write("Ingresa URL MJPEG (ej: http://192.168.1.100/video): ");
                videoPath = Console.ReadLine() ?? "";
                videoProcessPath = null;
                sourceType = VideoSourceType.MjpegOpenCV;
                break;
            case "5":
                if (!FFmpegHelper.IsAvailable)
                {
                    Console.WriteLine("\nFFmpeg no está disponible. Usa la opción 4 en su lugar.");
                    return;
                }
                Console.Write("Ingresa URL MJPEG (ej: http://192.168.1.100/video): ");
                videoPath = Console.ReadLine() ?? "";
                videoProcessPath = null;
                sourceType = VideoSourceType.MjpegFFmpeg;
                break;
            default:
                Console.WriteLine("Opción no válida");
                return;
        }

        Console.WriteLine($"\n=== Fuente seleccionada: {VideoSourceFactory.GetSourceDescription(sourceType.Value)} ===");
        Console.WriteLine("\n=== Selecciona modelo ===");
        Console.WriteLine("1. Procesar usando yolo11m 1 batch");
        Console.WriteLine("2. Procesar usando yolo11m 2 batch - two batch");
        Console.WriteLine("3. Procesar usando yolo11n 1 batch");
        Console.WriteLine("4. Procesar usando yolo11n 2 batch - two batch");
        Console.WriteLine("5. Procesar usando yolo11n 1 batch - Bytetrack");
        Console.WriteLine("6. Procesar usando yolo11n 2 batch - two batch - Bytetrack");
        Console.WriteLine("7. Procesar usando yolo26n 1 batch");
        Console.WriteLine("8. Procesar usando yolo26n 2 batch - two batch");
        Console.WriteLine("9. Procesar usando yolo26n 1 batch - Bytetrack");
        Console.WriteLine("10. Procesar usando yolo26n 2 batch - two batch - Bytetrack");

        Console.WriteLine("11. Salir");
        Console.Write("\nSelecciona una opción: ");

        string? opcion = Console.ReadLine();
        bool Yolo26 = false;

        switch (opcion)
        {
            case "1":
                usingYolo11m();
                break;
            case "2":
                usingYolo11m2batch();
                break;
            case "3":
                usingYolo11n1batch();
                break;
            case "4":
                usingYolo11n2batch();
                break;
            case "5":
                usingYolo11n1batchBytetrack();
                break;
            case "6":
                usingYolo11n2batchBytetrack();
                break;
            case "7":
                usingYolo26n1batch();
                Yolo26 = true;
                break;
            case "8":
                usingYolo26n2batch();
                Yolo26 = true;
                break;
            case "9":
                usingYolo26n1batchBytetrack();
                Yolo26 = true;
                break;
            case "10":
                usingYolo26n2batchBytetrack();
                Yolo26 = true;
                break;
            case "11":
                Console.WriteLine("Saliendo...");
                return;
            default:
                Console.WriteLine("Opción no válida");
                return;
        }

        // Cargar las clases del modelo seleccionado
        var availableClasses = LoadAndDisplayModelClasses(modelPath);

        // Seleccionar qué clases detectar
        HashSet<int> targetClasses = SelectTargetClasses(availableClasses);

        // Mostrar resumen de configuración
        DisplaySelectedClassesSummary(targetClasses, availableClasses);

        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║              INICIANDO DETECCIÓN                          ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
        Console.WriteLine($"\nModelo: {Path.GetFileName(modelPath)}");
        Console.WriteLine($"Video: {(sourceType == VideoSourceType.File ? Path.GetFileName(videoPath) : videoPath)}");
        Console.WriteLine($"Clases a detectar: {targetClasses.Count}");
        Console.WriteLine($"Modo: {(batch ? "2 Batch" : "1 Batch")} {(modelType.ToString().Contains("Bytetrack") ? "+ ByteTrack" : "")}");
        Console.WriteLine("\nPresiona cualquier tecla en la ventana de video para detener");
        Console.WriteLine("\nPresiona Enter para continuar...");
        Console.ReadLine();

        Capture Cap = new Capture(videoPath, videoProcessPath, modelPath, targetClasses, sourceType);

        switch (modelType) 
        {
            case ModelType.Yolo11:
                if (batch) 
                    Cap.runWithModel2Batch(modelType);
                else
                    Cap.runWithModel1Batch(modelType);
                break;
            case ModelType.Yolo11Bytetrack:
                if (batch)
                    Cap.runWithModel2BatchWithTracking(modelType);
                else
                    Cap.runWithModel1BatchWithTracking(modelType);
                break;
            case ModelType.Yolo26:
                if (batch)
                    Cap.runWithModel2BatchYolo26(modelType);
                else
                    Cap.runWithModel1BatchYolo26(modelType);
                break;
            case ModelType.Yolo26Bytetrack:
                if (batch)
                    Cap.runWithModel2BatchYolo26ByteTrack(modelType);
                else
                    Cap.runWithModel1BatchYolo26Bytetrack(modelType);
                break;
        }
    }
    private static void usingYolo11m()
    {
        modelPath = yolo11m;
        modelType = ModelType.Yolo11;
        Console.WriteLine("Usando modelo yolo11m");
    }
    private static void usingYolo11m2batch()
    {
        modelPath = yolo11m2batch;
        batch = true;
        modelType = ModelType.Yolo11;
        Console.WriteLine("Usando modelo yolo11m 2 batch");
    }
    private static void usingYolo11n1batch()
    {
        modelPath = yolo11n1batch;
        modelType = ModelType.Yolo11;
        Console.WriteLine("Usando modelo yolo11n 1 batch");
    }
    private static void usingYolo11n2batch()
    {
        modelPath = yolo11n2batch;
        batch = true;
        modelType = ModelType.Yolo11;
        Console.WriteLine("Usando modelo yolo11n 2 batch");
    }
    private static void usingYolo11n1batchBytetrack()
    {
        modelPath = yolo11n1batch;
        modelType = ModelType.Yolo11Bytetrack;
        Console.WriteLine("Usando modelo yolo11n 1 batch");
    }
    private static void usingYolo11n2batchBytetrack()
    {
        modelPath = yolo11n2batch;
        batch = true;
        modelType = ModelType.Yolo11Bytetrack;
        Console.WriteLine("Usando modelo yolo11n 2 batch");
    }
    private static void usingYolo26n1batch()
    {
        modelPath = yolo26n1batch;
        modelType = ModelType.Yolo26;
        Console.WriteLine("Usando modelo yolo26n 1 batch");
    }
    private static void usingYolo26n1batchBytetrack()
    {
        modelPath = yolo26n1batch;
        modelType = ModelType.Yolo26Bytetrack;
        Console.WriteLine("Usando modelo yolo26n 1 batch con ByteTrack");
    }
    private static void usingYolo26n2batch()
    {
        modelPath = yolo26n2batch;
        modelType = ModelType.Yolo26;
        batch = true;
        Console.WriteLine("Usando modelo yolo26n 2 batch");
    }
    private static void usingYolo26n2batchBytetrack()
    {
        modelPath = yolo26n2batch;
        modelType = ModelType.Yolo26Bytetrack;
        batch = true;
        Console.WriteLine("Usando modelo yolo26n 2 batch con ByteTrac");
    }

    /// <summary>
    /// Método para leer y mostrar las clases disponibles en un modelo ONNX
    /// </summary>
    private static Dictionary<int, string> LoadAndDisplayModelClasses(string modelPath)
    {
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║          LEYENDO INFORMACIÓN DEL MODELO                   ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝\n");

        var modelInfo = ModelMetadataReader.GetModelInfo(modelPath);

        if (!modelInfo.IsValid)
        {
            Console.WriteLine($"Error al cargar modelo: {modelInfo.ErrorMessage}");
            return new Dictionary<int, string>();
        }

        Console.WriteLine($"Modelo: {modelInfo.ModelName}");
        Console.WriteLine($"Tipo: {modelInfo.ModelType}");
        Console.WriteLine($"Total de clases: {modelInfo.Classes.Count}");

        if (!string.IsNullOrEmpty(modelInfo.ProducerName))
        {
            Console.WriteLine($"Productor: {modelInfo.ProducerName}");
        }

        Console.WriteLine("\nPrimeras clases disponibles:");
        int count = 0;
        foreach (var cls in modelInfo.Classes.OrderBy(x => x.Key).Take(15))
        {
            Console.Write($"  [{cls.Key,2}] {cls.Value,-20}");
            count++;
            if (count % 2 == 0)
                Console.WriteLine();
        }

        if (count % 2 != 0)
            Console.WriteLine();

        if (modelInfo.Classes.Count > 15)
        {
            Console.WriteLine($"\n  ... y {modelInfo.Classes.Count - 15} clases más");
        }

        return modelInfo.Classes;
    }

    /// <summary>
    /// Método para seleccionar clases de manera interactiva
    /// </summary>
    private static HashSet<int> SelectTargetClasses(Dictionary<int, string> availableClasses)
    {
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║          SELECCIÓN DE CLASES A DETECTAR                   ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");

        Console.WriteLine("\n1. Solo personas (clase 0) - Recomendado");
        Console.WriteLine("2. Vehículos (persona, bicicleta, carro, moto, autobús, camión)");
        Console.WriteLine("3. Animales (gato, perro, caballo, oveja, vaca, etc.)");
        Console.WriteLine("4. Ver todas las clases y seleccionar manualmente");
        Console.WriteLine("5. Todas las clases disponibles");
        Console.Write("\n Opción: ");

        string? option = Console.ReadLine();
        HashSet<int> targetClasses = new HashSet<int>();

        switch (option)
        {
            case "1":
                targetClasses.Add(0); // persona
                Console.WriteLine("\nSeleccionada: persona");
                break;

            case "2":
                int[] vehicles = { 0, 1, 2, 3, 5, 7 }; // persona, bicicleta, carro, moto, autobús, camión
                foreach (var id in vehicles)
                {
                    if (availableClasses.ContainsKey(id))
                    {
                        targetClasses.Add(id);
                        Console.WriteLine($"Agregada: [{id}] {availableClasses[id]}");
                    }
                }
                break;

            case "3":
                int[] animals = { 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }; // pájaro, gato, perro, caballo, oveja, vaca, elefante, oso, cebra, jirafa
                Console.WriteLine("\nClases de animales:");
                foreach (var id in animals)
                {
                    if (availableClasses.ContainsKey(id))
                    {
                        targetClasses.Add(id);
                        Console.WriteLine($"Agregada: [{id}] {availableClasses[id]}");
                    }
                }
                break;

            case "4":
                // Mostrar todas las clases disponibles
                Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
                Console.WriteLine("║              TODAS LAS CLASES DISPONIBLES                 ║");
                Console.WriteLine("╚═══════════════════════════════════════════════════════════╝\n");

                int columns = 2;
                int count = 0;
                var sortedClasses = availableClasses.OrderBy(x => x.Key).ToList();

                foreach (var cls in sortedClasses)
                {
                    Console.Write($"[{cls.Key,2}] {cls.Value,-25}");
                    count++;
                    if (count % columns == 0)
                        Console.WriteLine();
                }

                if (count % columns != 0)
                    Console.WriteLine();

                // Selección manual
                Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
                Console.WriteLine("║         INGRESA LOS IDs QUE DESEAS DETECTAR               ║");
                Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
                Console.WriteLine("\nEjemplos:");
                Console.WriteLine("  Para una sola clase: 0");
                Console.WriteLine("  Para múltiples clases: 0,1,2,15,16");
                Console.WriteLine("  Para un rango: 0-5 (detectará clases 0,1,2,3,4,5)");
                Console.Write("\nClases (separadas por coma): ");

                string? input = Console.ReadLine();

                if (!string.IsNullOrEmpty(input))
                {
                    foreach (var part in input.Split(','))
                    {
                        var trimmed = part.Trim();

                        // Verificar si es un rango (ej: 0-5)
                        if (trimmed.Contains('-'))
                        {
                            var range = trimmed.Split('-');
                            if (range.Length == 2 && 
                                int.TryParse(range[0].Trim(), out int start) && 
                                int.TryParse(range[1].Trim(), out int end))
                            {
                                for (int i = start; i <= end; i++)
                                {
                                    if (availableClasses.ContainsKey(i))
                                    {
                                        targetClasses.Add(i);
                                    }
                                }
                            }
                        }
                        // Si es un ID individual
                        else if (int.TryParse(trimmed, out int classId) && 
                                availableClasses.ContainsKey(classId))
                        {
                            targetClasses.Add(classId);
                        }
                    }

                    Console.WriteLine("\nClases seleccionadas:");
                    foreach (var id in targetClasses.OrderBy(x => x))
                    {
                        Console.WriteLine($"  [{id}] {availableClasses[id]}");
                    }
                }
                break;

            case "5":
                foreach (var cls in availableClasses)
                {
                    targetClasses.Add(cls.Key);
                }
                Console.WriteLine($"\nSeleccionadas TODAS las {targetClasses.Count} clases");
                break;

            default:
                targetClasses.Add(0); // Solo personas por defecto
                Console.WriteLine("\nOpción no válida, usando clase 0 (persona) por defecto");
                break;
        }

        if (targetClasses.Count == 0)
        {
            Console.WriteLine("\nNo se seleccionaron clases válidas, usando clase 0 (persona) por defecto");
            targetClasses.Add(0);
        }

        return targetClasses;
    }

    /// <summary>
    /// Método auxiliar para mostrar un resumen de las clases seleccionadas
    /// </summary>
    private static void DisplaySelectedClassesSummary(HashSet<int> targetClasses, Dictionary<int, string> availableClasses)
    {
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║              RESUMEN DE CONFIGURACIÓN                     ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
        Console.WriteLine($"\nTotal de clases a detectar: {targetClasses.Count}");
        Console.WriteLine("\nClases activas:");

        foreach (var id in targetClasses.OrderBy(x => x).Take(10))
        {
            if (availableClasses.ContainsKey(id))
            {
                Console.WriteLine($"   [{id}] {availableClasses[id]}");
            }
        }

        if (targetClasses.Count > 10)
        {
            Console.WriteLine($"   ... y {targetClasses.Count - 10} clases más");
        }
    }
}