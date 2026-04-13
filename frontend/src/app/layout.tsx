import type { Metadata } from "next"
import { InferenceForm } from "@/components/inference-form"
import "./globals.css"

export const metadata: Metadata = {
  title: "YOLOv8 Inference Dashboard",
  description: "Real-time object detection with acceleration benchmarking",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-foreground">
          {children}
        </div>
      </body>
    </html>
  )
}
