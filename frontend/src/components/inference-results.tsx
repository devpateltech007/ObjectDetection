"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { InferenceResponse } from "@/lib/api"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { AlertCircle, CheckCircle2 } from "lucide-react"

interface InferenceResultsProps {
  results: InferenceResponse
  imagePreviewUrl?: string | null
}

export function InferenceResults({ results, imagePreviewUrl }: InferenceResultsProps) {
  const [imgSize, setImgSize] = useState<{ width: number; height: number }>({ width: 1, height: 1 })
  const acceleration = results.acceleration ?? []

  // Prepare acceleration comparison data
  const speedData = [
    { name: "Baseline", latency: results.speed.latency_ms, fps: results.speed.fps },
    ...acceleration.map((acc) => ({
      name: acc.engine,
      latency: acc.speed?.latency_ms || 0,
      fps: acc.speed?.fps || 0,
    })),
  ]

  // Prepare accuracy comparison data
  const accuracyData = [
    { name: "Baseline", accuracy: results.evaluation.accuracy || 0, map50: results.evaluation.map50 || 0 },
    ...acceleration.map((acc) => ({
      name: acc.engine,
      accuracy: acc.evaluation?.accuracy || 0,
      map50: acc.evaluation?.map50 || 0,
    })),
  ]

  // Prepare metrics comparison
  const metricsComparison = [
    {
      engine: "Baseline",
      tp: results.evaluation.tp,
      fp: results.evaluation.fp,
      fn: results.evaluation.fn,
    },
    ...acceleration
      .filter((acc) => acc.available)
      .map((acc) => ({
        engine: acc.engine,
        tp: acc.evaluation?.tp || 0,
        fp: acc.evaluation?.fp || 0,
        fn: acc.evaluation?.fn || 0,
      })),
  ]

  return (
    <div className="space-y-6">
      {imagePreviewUrl && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Uploaded Image</CardTitle>
              <CardDescription>Original image used for inference</CardDescription>
            </CardHeader>
            <CardContent>
              <img
                src={imagePreviewUrl}
                alt="Uploaded"
                className="w-full rounded-md border"
                onLoad={(e) => {
                  const target = e.currentTarget
                  setImgSize({ width: target.naturalWidth || 1, height: target.naturalHeight || 1 })
                }}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Detected Image</CardTitle>
              <CardDescription>Predicted bounding boxes overlaid on uploaded image</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative w-full rounded-md border overflow-hidden">
                <img src={imagePreviewUrl} alt="Detected" className="w-full" />
                {results.detections.map((det, idx) => {
                  const [x1, y1, x2, y2] = det.bbox
                  const left = Math.max(0, (x1 / imgSize.width) * 100)
                  const top = Math.max(0, (y1 / imgSize.height) * 100)
                  const width = Math.max(0, ((x2 - x1) / imgSize.width) * 100)
                  const height = Math.max(0, ((y2 - y1) / imgSize.height) * 100)

                  return (
                    <div
                      key={idx}
                      className="absolute border-2 border-green-400"
                      style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                    >
                      <div className="absolute -top-6 left-0 bg-green-500 text-white text-[10px] px-1 py-0.5 rounded">
                        {det.label} {(det.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Detections</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.num_detections}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Baseline Latency</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.speed.latency_ms.toFixed(0)}ms</div>
            <p className="text-xs text-muted-foreground">{results.speed.fps.toFixed(2)} FPS</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {results.evaluation.accuracy !== null ? (results.evaluation.accuracy * 100).toFixed(0) + "%" : "N/A"}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">mAP50</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {results.evaluation.map50 !== null ? (results.evaluation.map50 * 100).toFixed(0) + "%" : "N/A"}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Results Tabs */}
      <Tabs defaultValue="speed" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="speed">Speed</TabsTrigger>
          <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
          <TabsTrigger value="detections">Detections</TabsTrigger>
          <TabsTrigger value="engines">Engines</TabsTrigger>
        </TabsList>

        {/* Speed Comparison */}
        <TabsContent value="speed">
          <Card>
            <CardHeader>
              <CardTitle>Latency & FPS Comparison</CardTitle>
              <CardDescription>Inference speed across different acceleration engines</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={speedData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis yAxisId="left" label={{ value: "Latency (ms)", angle: -90, position: "insideLeft" }} />
                  <YAxis yAxisId="right" orientation="right" label={{ value: "FPS", angle: 90, position: "insideRight" }} />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="latency" fill="#ef4444" name="Latency (ms)" />
                  <Bar yAxisId="right" dataKey="fps" fill="#22c55e" name="FPS" />
                </BarChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  { engine: "Baseline", metrics: results.speed },
                  ...acceleration.map((acc) => ({ engine: acc.engine, metrics: acc.speed })),
                ].map((item, idx) => (
                  <Card key={idx} className="bg-muted/50">
                    <CardContent className="pt-6">
                      <h4 className="font-semibold mb-4">{item.engine}</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Latency:</span>
                          <span className="font-mono">
                            {item.metrics?.latency_ms !== undefined ? `${item.metrics.latency_ms.toFixed(2)}ms` : "N/A"}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">FPS:</span>
                          <span className="font-mono">
                            {item.metrics?.fps !== undefined ? item.metrics.fps.toFixed(2) : "N/A"}
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Accuracy Comparison */}
        <TabsContent value="accuracy">
          <Card>
            <CardHeader>
              <CardTitle>Accuracy & mAP Comparison</CardTitle>
              <CardDescription>Detection quality across engines</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={accuracyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis label={{ value: "Score", angle: -90, position: "insideLeft" }} domain={[0, 1]} />
                  <Tooltip formatter={(value) => (typeof value === "number" ? (value * 100).toFixed(2) + "%" : value)} />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
                  <Bar dataKey="map50" fill="#8b5cf6" name="mAP50" />
                </BarChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  { engine: "Baseline", eval: results.evaluation },
                  ...acceleration.map((a) => ({ engine: a.engine, eval: a.evaluation })),
                ].map((item, idx) => (
                  <Card key={idx} className="bg-muted/50">
                    <CardContent className="pt-6">
                      <h4 className="font-semibold mb-4">{item.engine}</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Accuracy:</span>
                          <span className="font-mono">{item.eval.accuracy !== null ? (item.eval.accuracy * 100).toFixed(2) + "%" : "N/A"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">mAP50:</span>
                          <span className="font-mono">{item.eval.map50 !== null ? (item.eval.map50 * 100).toFixed(2) + "%" : "N/A"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Precision:</span>
                          <span className="font-mono">{item.eval.precision !== null ? (item.eval.precision * 100).toFixed(2) + "%" : "N/A"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Recall:</span>
                          <span className="font-mono">{item.eval.recall !== null ? (item.eval.recall * 100).toFixed(2) + "%" : "N/A"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">F1:</span>
                          <span className="font-mono">{item.eval.f1 !== null ? item.eval.f1.toFixed(4) : "N/A"}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Detections */}
        <TabsContent value="detections">
          <Card>
            <CardHeader>
              <CardTitle>Detected Objects</CardTitle>
              <CardDescription>Objects detected by the model in baseline</CardDescription>
            </CardHeader>
            <CardContent>
              {results.detections.length === 0 ? (
                <p className="text-muted-foreground text-sm">No objects detected</p>
              ) : (
                <div className="space-y-3">
                  {results.detections.map((det, idx) => (
                    <div key={idx} className="flex items-start justify-between p-3 bg-muted rounded-lg">
                      <div className="space-y-1">
                        <p className="font-medium">{det.label}</p>
                        <p className="text-xs text-muted-foreground">
                          Confidence: {(det.confidence * 100).toFixed(2)}%
                        </p>
                        <p className="text-xs text-muted-foreground font-mono">
                          Box: [{Math.round(det.bbox[0])}, {Math.round(det.bbox[1])}, {Math.round(det.bbox[2])}, {Math.round(det.bbox[3])}]
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1 text-green-600">
                          <CheckCircle2 className="h-4 w-4" />
                          <span className="text-xs">Detected</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Engine Details */}
        <TabsContent value="engines">
          <Card>
            <CardHeader>
              <CardTitle>Acceleration Engines Status</CardTitle>
              <CardDescription>Availability and details of each inference engine</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {acceleration.map((engine, idx) => (
                <div key={idx} className="border rounded-lg p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold">{engine.engine}</h4>
                    {engine.available ? (
                      <div className="flex items-center gap-2 text-green-600">
                        <CheckCircle2 className="h-4 w-4" />
                        <span className="text-sm">Available</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-amber-600">
                        <AlertCircle className="h-4 w-4" />
                        <span className="text-sm">Unavailable</span>
                      </div>
                    )}
                  </div>

                  {engine.available ? (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                      <div>
                        <p className="text-muted-foreground text-xs">Latency</p>
                        <p className="font-mono font-semibold">{engine.speed?.latency_ms.toFixed(2)}ms</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground text-xs">FPS</p>
                        <p className="font-mono font-semibold">{engine.speed?.fps.toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground text-xs">Detections</p>
                        <p className="font-mono font-semibold">{engine.detections.length}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground text-xs">True Positives</p>
                        <p className="font-mono font-semibold">{engine.evaluation.tp}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-amber-700 bg-amber-50 dark:bg-amber-950 p-3 rounded">
                      <p className="font-mono text-xs">{engine.error}</p>
                    </div>
                  )}

                  {engine.available && engine.artifact && (
                    <p className="text-xs text-muted-foreground font-mono break-all">
                      Artifact: {engine.artifact.split("\\").pop()}
                    </p>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
