import { InferenceForm } from "@/components/inference-form"
import { Zap, Gauge, Target, Cpu } from "lucide-react"

export default function Home() {
  return (
    <main className="container mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-12">
        <div className="max-w-3xl mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Inference Dashboard
            </h1>
            </div>
          <p className="text-lg text-slate-300 mb-8">
            Compare YOLOv8n & RF-DETR across baseline, OpenVINO & ONNX Runtime CUDA
          </p>

        </div>
      </div>

      {/* Inference Form */}
      <div className="max-w-4xl mx-auto">
        <InferenceForm />
      </div>

    </main>
  )
}
