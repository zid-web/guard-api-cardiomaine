import { useState, useRef, useCallback } from "react";
import { Mic, MicOff, Upload, FileText, Loader2, CheckCircle2, AlertTriangle } from "lucide-react";

/**
 * Panneau combiné :
 * - Zone de drag & drop pour uploader un PDF de planning
 * - Bouton micro pour dicter une consigne ("demain S remplace B en garde")
 *
 * Props :
 * - apiBaseUrl : URL de base de l'API FastAPI (ex: "https://guard-api-cardiomaine.onrender.com")
 * - apiKey : clé API (header x-api-key), optionnel
 * - currentWeekRequest : objet GenerateWeekRequest actuel (nécessaire pour appliquer la commande vocale)
 * - knownDoctors : liste des codes médecins valides, ex: ["W","O","M","A","Z","CH","FV"]
 * - onScheduleUpdated : callback(updatedSchedule) appelé quand le planning est recalculé
 * - onPdfParsed : callback(extractedData) appelé quand un PDF a été traité côté backend
 */
export default function VoiceAndUploadPanel({
  apiBaseUrl,
  apiKey = "",
  currentWeekRequest,
  knownDoctors = [],
  onScheduleUpdated,
  onPdfParsed,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState(null); // { type: "loading"|"success"|"error", message: string }
  const recognitionRef = useRef(null);
  const fileInputRef = useRef(null);

  // ---------------------------------------------------------
  // Reconnaissance vocale (Web Speech API)
  // ---------------------------------------------------------
  const startListening = useCallback(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setStatus({
        type: "error",
        message:
          "La reconnaissance vocale n'est pas disponible dans ce navigateur. Utilisez Chrome ou Edge.",
      });
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "fr-FR";
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onstart = () => {
      setIsListening(true);
      setTranscript("");
      setStatus(null);
    };

    recognition.onresult = (event) => {
      const text = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join(" ");
      setTranscript(text);
    };

    recognition.onerror = (event) => {
      setIsListening(false);
      setStatus({ type: "error", message: `Erreur micro : ${event.error}` });
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, []);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setIsListening(false);
  }, []);

  const sendVoiceCommand = useCallback(async () => {
    if (!transcript.trim()) return;
    setStatus({ type: "loading", message: "Interprétation de la consigne..." });

    try {
      const res = await fetch(`${apiBaseUrl}/voice-command`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
        },
        body: JSON.stringify({
          text: transcript,
          reference_date: new Date().toISOString().slice(0, 10),
          known_doctors: knownDoctors,
          current_week_request: currentWeekRequest,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setStatus({
          type: "error",
          message: data.detail || "Erreur lors du traitement de la consigne.",
        });
        return;
      }

      setStatus({ type: "success", message: data.message });
      onScheduleUpdated?.(data.updated_schedule);
      setTranscript("");
    } catch (err) {
      setStatus({ type: "error", message: `Erreur réseau : ${err.message}` });
    }
  }, [transcript, apiBaseUrl, apiKey, knownDoctors, currentWeekRequest, onScheduleUpdated]);

  // ---------------------------------------------------------
  // Upload PDF (drag & drop + sélection manuelle)
  // ---------------------------------------------------------
  const uploadPdf = useCallback(
    async (file) => {
      if (!file || file.type !== "application/pdf") {
        setStatus({ type: "error", message: "Merci de déposer un fichier PDF." });
        return;
      }

      setStatus({ type: "loading", message: `Analyse de ${file.name}...` });

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch(`${apiBaseUrl}/upload-planning-pdf`, {
          method: "POST",
          headers: { "x-api-key": apiKey },
          body: formData,
        });

        const data = await res.json();

        if (!res.ok) {
          setStatus({
            type: "error",
            message: data.detail || "Erreur lors de l'analyse du PDF.",
          });
          return;
        }

        setStatus({ type: "success", message: `${file.name} analysé avec succès.` });
        onPdfParsed?.(data);
      } catch (err) {
        setStatus({ type: "error", message: `Erreur réseau : ${err.message}` });
      }
    },
    [apiBaseUrl, apiKey, onPdfParsed]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      uploadPdf(file);
    },
    [uploadPdf]
  );

  const handleFileSelect = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      uploadPdf(file);
    },
    [uploadPdf]
  );

  return (
    <div className="w-full max-w-xl mx-auto space-y-4">
      {/* Zone drag & drop PDF */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors
          ${isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={handleFileSelect}
          className="hidden"
        />
        <Upload className="mx-auto mb-2 text-gray-400" size={28} />
        <p className="text-sm text-gray-600">
          Glissez-déposez le PDF du planning ici, ou cliquez pour parcourir
        </p>
      </div>

      {/* Commande vocale */}
      <div className="border rounded-xl p-4 space-y-3">
        <div className="flex items-center gap-3">
          <button
            onClick={isListening ? stopListening : startListening}
            className={`flex items-center justify-center w-12 h-12 rounded-full transition-colors
              ${isListening ? "bg-red-500 text-white animate-pulse" : "bg-gray-100 hover:bg-gray-200 text-gray-700"}`}
            aria-label={isListening ? "Arrêter l'écoute" : "Dicter une consigne"}
          >
            {isListening ? <MicOff size={20} /> : <Mic size={20} />}
          </button>
          <div className="flex-1 text-sm text-gray-600">
            {isListening
              ? "Je vous écoute... (ex: \"demain S remplace B en garde\")"
              : "Cliquez sur le micro pour dicter une modification du planning"}
          </div>
        </div>

        {transcript && (
          <div className="bg-gray-50 rounded-lg p-3 text-sm text-gray-800 flex items-start gap-2">
            <FileText size={16} className="mt-0.5 shrink-0 text-gray-400" />
            <span>{transcript}</span>
          </div>
        )}

        {transcript && !isListening && (
          <button
            onClick={sendVoiceCommand}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-2 text-sm font-medium transition-colors"
          >
            Valider et appliquer au planning
          </button>
        )}
      </div>

      {/* Statut */}
      {status && (
        <div
          className={`flex items-center gap-2 text-sm rounded-lg p-3
            ${status.type === "error" ? "bg-red-50 text-red-700" : ""}
            ${status.type === "success" ? "bg-green-50 text-green-700" : ""}
            ${status.type === "loading" ? "bg-blue-50 text-blue-700" : ""}`}
        >
          {status.type === "loading" && <Loader2 size={16} className="animate-spin" />}
          {status.type === "success" && <CheckCircle2 size={16} />}
          {status.type === "error" && <AlertTriangle size={16} />}
          <span>{status.message}</span>
        </div>
      )}
    </div>
  );
}
