"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import {
  runInferenceUpload,
  runVideoInferenceUpload,
  type InferenceResponse,
  type GroundTruthBox,
  type VideoInferenceResponse,
} from "@/lib/api"
import { InferenceResults } from "./inference-results"
import { VideoInferenceResults } from "./video-inference-results"
import { Loader2 } from "lucide-react"

interface GroundTruthDraft {
  label: string
  x1: string
  y1: string
  x2: string
  y2: string
}

export function InferenceForm() {
  const [modelName, setModelName] = useState<"yolov8n" | "rf-detr">("yolov8n")
  const [includeGroundTruth, setIncludeGroundTruth] = useState(false)
  const [groundTruthItems, setGroundTruthItems] = useState<GroundTruthDraft[]>([
    { label: "bottle", x1: "408", y1: "103", x2: "586", y2: "655" },
  ])
  const [loading, setLoading] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedFilePreviewUrl, setSelectedFilePreviewUrl] = useState<string | null>(null)
  const [lastRunImagePreviewUrl, setLastRunImagePreviewUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [results, setResults] = useState<InferenceResponse | null>(null)
  const [videoResults, setVideoResults] = useState<VideoInferenceResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Revoke only on component unmount; revoking on every state change can break images in use.
    return () => {
      if (selectedFilePreviewUrl) {
        URL.revokeObjectURL(selectedFilePreviewUrl)
      }
      if (lastRunImagePreviewUrl && lastRunImagePreviewUrl !== selectedFilePreviewUrl) {
        URL.revokeObjectURL(lastRunImagePreviewUrl)
      }
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!selectedFile) {
      setError("Please upload an image first.")
      return
    }

    setLoading(true)
    setError(null)

    try {
      let groundTruth: GroundTruthBox[] | undefined
      const isVideo = selectedFile.type.startsWith("video/")

      if (includeGroundTruth && !isVideo) {
        const parsed = groundTruthItems
          .filter((item) => item.label.trim().length > 0)
          .map((item) => {
            const x1 = Number(item.x1)
            const y1 = Number(item.y1)
            const x2 = Number(item.x2)
            const y2 = Number(item.y2)
            return {
              label: item.label.trim(),
              bbox: [x1, y1, x2, y2] as [number, number, number, number],
            }
          })

        const hasInvalid = parsed.some(
          (item) => item.bbox.some((v) => !Number.isFinite(v)) || item.bbox[2] <= item.bbox[0] || item.bbox[3] <= item.bbox[1]
        )

        if (parsed.length === 0 || hasInvalid) {
          setError("Ground truth requires at least one valid object with label and bbox where x2>x1 and y2>y1.")
          setLoading(false)
          return
        }

        groundTruth = parsed
      }

      if (isVideo) {
        const response = await runVideoInferenceUpload(selectedFile, modelName)
        if (lastRunImagePreviewUrl) {
          URL.revokeObjectURL(lastRunImagePreviewUrl)
        }
        setVideoResults(response)
        setResults(null)
        setLastRunImagePreviewUrl(null)
      } else {
        if (lastRunImagePreviewUrl && lastRunImagePreviewUrl !== selectedFilePreviewUrl) {
          URL.revokeObjectURL(lastRunImagePreviewUrl)
        }
        const response = await runInferenceUpload(selectedFile, modelName, groundTruth)
        setLastRunImagePreviewUrl(selectedFilePreviewUrl)
        setResults(response)
        setVideoResults(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed")
    } finally {
      setLoading(false)
    }
  }

  const updateGroundTruthItem = (index: number, key: keyof GroundTruthDraft, value: string) => {
    setGroundTruthItems((prev) =>
      prev.map((item, idx) => (idx === index ? { ...item, [key]: value } : item))
    )
  }

  const addGroundTruthItem = () => {
    setGroundTruthItems((prev) => [...prev, { label: "", x1: "", y1: "", x2: "", y2: "" }])
  }

  const removeGroundTruthItem = (index: number) => {
    setGroundTruthItems((prev) => prev.filter((_, idx) => idx !== index))
  }

  const handleFileSelect = (file: File | null) => {
    if (!file) {
      return
    }
    if (!file.type.startsWith("image/") && !file.type.startsWith("video/")) {
      setError("Please upload a valid image or video file.")
      return
    }
    setError(null)
    if (selectedFilePreviewUrl && selectedFilePreviewUrl !== lastRunImagePreviewUrl) {
      URL.revokeObjectURL(selectedFilePreviewUrl)
    }
    setSelectedFilePreviewUrl(URL.createObjectURL(file))
    setSelectedFile(file)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Inference Request</CardTitle>
          <CardDescription>Configure your inference parameters</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Image Path */}
            {/* <div className="space-y-2">
              <Label htmlFor="image-path">Image Path</Label>
              <Input
                id="image-path"
                placeholder="e.g., img/test.jpeg"
                value={imagePath}
                onChange={(e) => setImagePath(e.target.value)}
                disabled={!!selectedFile}
              />
              <p className="text-xs text-muted-foreground">
                Path-based inference is used when no uploaded file is selected.
              </p>
            </div> */}

            <div className="space-y-2">
              <Label>Choose or Drag & Drop Image/Video</Label>
              <div
                className={`rounded-md border-2 border-dashed p-6 text-center transition-colors ${
                  isDragging ? "border-primary bg-primary/5" : "border-input"
                }`}
                onDragOver={(e) => {
                  e.preventDefault()
                  setIsDragging(true)
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setIsDragging(false)
                  handleFileSelect(e.dataTransfer.files?.[0] ?? null)
                }}
              >
                <p className="text-sm text-muted-foreground">Drop image/video here, or select from disk</p>
                <Input
                  type="file"
                  accept="image/*,video/*"
                  className="mt-3"
                  onChange={(e) => handleFileSelect(e.target.files?.[0] ?? null)}
                />
                {!selectedFile && (
                  <p className="mt-3 text-xs text-amber-600">Please upload an image or video first to run inference.</p>
                )}
                {selectedFile && (
                  <div className="mt-3 flex items-center justify-between rounded bg-muted px-3 py-2 text-xs">
                    <span className="truncate">Selected: {selectedFile.name} ({selectedFile.type || "unknown"})</span>
                    <button
                      type="button"
                      className="text-destructive"
                      onClick={() => {
                        setSelectedFile(null)
                        if (selectedFilePreviewUrl && selectedFilePreviewUrl !== lastRunImagePreviewUrl) {
                          URL.revokeObjectURL(selectedFilePreviewUrl)
                        }
                        setSelectedFilePreviewUrl(null)
                      }}
                    >
                      Remove
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Model Selection */}
            <div className="space-y-3">
              <Label>Model Selection</Label>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="yolov8n"
                    checked={modelName === "yolov8n"}
                    onChange={() => setModelName("yolov8n")}
                  />
                  <Label htmlFor="yolov8n" className="font-normal cursor-pointer">
                    YOLOv8n
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="rf-detr"
                    checked={modelName === "rf-detr"}
                    onChange={() => setModelName("rf-detr")}
                  />
                  <Label htmlFor="rf-detr" className="font-normal cursor-pointer">
                    RF-DETR
                  </Label>
                </div>
              </div>
            </div>

            {/* Ground Truth */}
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="include-gt"
                  checked={includeGroundTruth}
                  disabled={selectedFile?.type.startsWith("video/")}
                  onChange={(e) => setIncludeGroundTruth(e.target.checked)}
                />
                <Label htmlFor="include-gt" className="font-normal cursor-pointer">
                  Include Ground Truth (for evaluation metrics) (disabled for video)
                </Label>
              </div>

              {includeGroundTruth && (
                <div className="space-y-4 pl-6 border-l-2 border-muted">
                  {groundTruthItems.map((item, idx) => (
                    <div key={idx} className="rounded-md border border-input p-3 space-y-3">
                      <div className="flex items-center justify-between">
                        <Label>Object {idx + 1}</Label>
                        {groundTruthItems.length > 1 && (
                          <Button type="button" variant="outline" size="sm" onClick={() => removeGroundTruthItem(idx)}>
                            Remove
                          </Button>
                        )}
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor={`gt-label-${idx}`}>Label</Label>
                        <Input
                          id={`gt-label-${idx}`}
                          placeholder="e.g., bottle"
                          value={item.label}
                          onChange={(e) => updateGroundTruthItem(idx, "label", e.target.value)}
                        />
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        <Input
                          placeholder="x1"
                          value={item.x1}
                          onChange={(e) => updateGroundTruthItem(idx, "x1", e.target.value)}
                        />
                        <Input
                          placeholder="y1"
                          value={item.y1}
                          onChange={(e) => updateGroundTruthItem(idx, "y1", e.target.value)}
                        />
                        <Input
                          placeholder="x2"
                          value={item.x2}
                          onChange={(e) => updateGroundTruthItem(idx, "x2", e.target.value)}
                        />
                        <Input
                          placeholder="y2"
                          value={item.y2}
                          onChange={(e) => updateGroundTruthItem(idx, "y2", e.target.value)}
                        />
                      </div>
                    </div>
                  ))}

                  <Button type="button" variant="outline" onClick={addGroundTruthItem}>
                    Add Object
                  </Button>
                </div>
              )}
            </div>

            {error && (
              <div className="p-3 bg-destructive/10 border border-destructive/30 rounded-md text-sm text-destructive">
                {error}
              </div>
            )}

            <Button
              type="submit"
              disabled={loading || !selectedFile}
              className="w-full bg-blue-600 text-white font-semibold border border-blue-700 shadow-md hover:bg-blue-700 disabled:bg-slate-400 disabled:border-slate-500"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running Inference...
                </>
              ) : (
                "Run Inference"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {results && <InferenceResults results={results} imagePreviewUrl={lastRunImagePreviewUrl} />}
      {videoResults && <VideoInferenceResults results={videoResults} />}
    </div>
  )
}
