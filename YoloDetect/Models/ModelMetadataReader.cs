using Microsoft.ML.OnnxRuntime;
using System.Text;
using System.IO;

namespace YoloDetect.Models
{
    public class ModelMetadataReader
    {
        /// <summary>
        /// Lee las clases desde el metadata del modelo ONNX
        /// </summary>
        public static Dictionary<int, string> ReadClassesFromModel(string modelPath)
        {
            var classes = new Dictionary<int, string>();

            try
            {
                using var session = new InferenceSession(modelPath);
                var metadata = session.ModelMetadata;

                // Intentar leer desde diferentes formatos de metadata
                // Formato 1: "names" como JSON o string separado por comas
                if (metadata.CustomMetadataMap.TryGetValue("names", out string? namesValue))
                {
                    classes = ParseNamesMetadata(namesValue);
                    if (classes.Count > 0)
                        return classes;
                }

                // Formato 2: Clases individuales como "class_0", "class_1", etc.
                foreach (var kvp in metadata.CustomMetadataMap)
                {
                    if (kvp.Key.StartsWith("class_"))
                    {
                        if (int.TryParse(kvp.Key.Substring(6), out int classId))
                        {
                            classes[classId] = kvp.Value;
                        }
                    }
                }

                if (classes.Count > 0)
                    return classes;

                // Formato 3: "labels" o "class_names"
                if (metadata.CustomMetadataMap.TryGetValue("labels", out string? labelsValue))
                {
                    classes = ParseNamesMetadata(labelsValue);
                    if (classes.Count > 0)
                        return classes;
                }

                if (metadata.CustomMetadataMap.TryGetValue("class_names", out string? classNamesValue))
                {
                    classes = ParseNamesMetadata(classNamesValue);
                    if (classes.Count > 0)
                        return classes;
                }

                Console.WriteLine($"No se encontraron clases en el metadata del modelo. Usando clases COCO por defecto.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error leyendo metadata del modelo: {ex.Message}. Usando clases COCO por defecto.");
            }

            return classes;
        }

        /// <summary>
        /// Parsea el string de nombres desde el metadata
        /// Soporta formato JSON: {"0": "person", "1": "bicycle"...}
        /// o formato simple: person,bicycle,car...
        /// </summary>
        private static Dictionary<int, string> ParseNamesMetadata(string namesValue)
        {
            var classes = new Dictionary<int, string>();

            try
            {
                // Intentar parsear como JSON
                if (namesValue.TrimStart().StartsWith("{"))
                {
                    // Parseo manual simple de JSON
                    var cleaned = namesValue.Trim('{', '}', ' ');
                    var pairs = cleaned.Split(',');

                    foreach (var pair in pairs)
                    {
                        var parts = pair.Split(':');
                        if (parts.Length == 2)
                        {
                            var keyStr = parts[0].Trim().Trim('"', '\'', ' ');
                            var valueStr = parts[1].Trim().Trim('"', '\'', ' ');

                            if (int.TryParse(keyStr, out int classId))
                            {
                                classes[classId] = valueStr;
                            }
                        }
                    }
                }
                // Si es una lista separada por comas
                else if (namesValue.Contains(',') || namesValue.Contains('['))
                {
                    var cleaned = namesValue.Trim('[', ']', ' ');
                    var items = cleaned.Split(',');

                    for (int i = 0; i < items.Length; i++)
                    {
                        var className = items[i].Trim().Trim('"', '\'', ' ');
                        if (!string.IsNullOrWhiteSpace(className))
                        {
                            classes[i] = className;
                        }
                    }
                }
                // Si es una sola palabra, asumir que es clase 0
                else if (!string.IsNullOrWhiteSpace(namesValue))
                {
                    classes[0] = namesValue.Trim().Trim('"', '\'', ' ');
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error parseando metadata de clases: {ex.Message}");
            }

            return classes;
        }

        /// <summary>
        /// Obtiene información detallada del modelo ONNX
        /// </summary>
        public static ModelInfo GetModelInfo(string modelPath)
        {
            var info = new ModelInfo
            {
                ModelPath = modelPath,
                ModelName = Path.GetFileNameWithoutExtension(modelPath)
            };

            try
            {
                using var session = new InferenceSession(modelPath);
                var metadata = session.ModelMetadata;

                info.ProducerName = metadata.ProducerName;
                info.Version = metadata.Version.ToString();
                info.Description = metadata.Description;

                // Obtener información de inputs
                foreach (var input in session.InputMetadata)
                {
                    info.Inputs.Add($"{input.Key}: {string.Join("x", input.Value.Dimensions)}");
                }

                // Obtener información de outputs
                foreach (var output in session.OutputMetadata)
                {
                    info.Outputs.Add($"{output.Key}: {string.Join("x", output.Value.Dimensions)}");
                }

                // Detectar tipo de modelo basado en outputs
                if (session.OutputMetadata.Count > 0)
                {
                    var firstOutput = session.OutputMetadata.First();
                    var dims = firstOutput.Value.Dimensions;
                    
                    // YOLOv8/v11 típicamente tiene salida [1, 84, 8400] o similar
                    // YOLO26 puede tener formato diferente
                    if (dims.Length >= 3)
                    {
                        info.ModelType = dims[1] > 100 ? "YOLO v8/v11" : "YOLO v26 o personalizado";
                    }
                }

                info.Classes = ReadClassesFromModel(modelPath);
                info.IsValid = true;
            }
            catch (Exception ex)
            {
                info.IsValid = false;
                info.ErrorMessage = ex.Message;
                Console.WriteLine($"Error cargando modelo: {ex.Message}");
            }

            return info;
        }
    }

    public class ModelInfo
    {
        public string ModelPath { get; set; } = string.Empty;
        public string ModelName { get; set; } = string.Empty;
        public string ProducerName { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string ModelType { get; set; } = "Desconocido";
        public List<string> Inputs { get; set; } = new();
        public List<string> Outputs { get; set; } = new();
        public Dictionary<int, string> Classes { get; set; } = new();
        public bool IsValid { get; set; }
        public string ErrorMessage { get; set; } = string.Empty;

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Modelo: {ModelName}");
            sb.AppendLine($"Tipo: {ModelType}");
            sb.AppendLine($"Clases: {Classes.Count}");
            if (!string.IsNullOrEmpty(ProducerName))
                sb.AppendLine($"Productor: {ProducerName}");
            return sb.ToString();
        }
    }
}
