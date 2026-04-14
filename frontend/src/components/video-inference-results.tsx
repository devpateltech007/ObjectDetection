"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { VideoInferenceResponse } from "@/lib/api"
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

interface VideoInferenceResultsProps {
  results: VideoInferenceResponse
}

export function VideoInferenceResults({ results }: VideoInferenceResultsProps) {
  const accelerationSummary = results.acceleration_summary ?? []
  const speedData = [
    { name: "baseline", latency: results.avg_latency_ms, fps: results.avg_fps },
    ...accelerationSummary.map((engine) => ({
      name: engine.engine,
      latency: engine.avg_latency_ms ?? 0,
      fps: engine.avg_fps ?? 0,
    })),
  ]

  const accuracyData = [
    {
      name: "baseline",
      accuracy: results.evaluation?.accuracy ?? 0,
      map50: results.evaluation?.map50 ?? 0,
    },
    ...accelerationSummary.map((engine) => ({
      name: engine.engine,
      accuracy: engine.evaluation?.accuracy ?? 0,
      map50: engine.evaluation?.map50 ?? 0,
    })),
  ]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Processed Frames</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.processed_frames}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Avg Latency</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.avg_latency_ms.toFixed(1)}ms</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Avg FPS</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.avg_fps.toFixed(2)}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Video FPS</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.video_fps.toFixed(2)}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Video Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {results.evaluation?.accuracy !== null && results.evaluation?.accuracy !== undefined
                ? `${(results.evaluation.accuracy * 100).toFixed(0)}%`
                : "N/A"}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Video mAP50</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {results.evaluation?.map50 !== null && results.evaluation?.map50 !== undefined
                ? `${(results.evaluation.map50 * 100).toFixed(0)}%`
                : "N/A"}
            </div>
          </CardContent>
        </Card>
      </div>

      {accelerationSummary.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Acceleration Summary (Video)</CardTitle>
            <CardDescription>
              Aggregated speed and quality metrics for accelerated engines across sampled frames.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h4 className="mb-2 text-sm font-medium text-muted-foreground">Latency and FPS Comparison</h4>
                <ResponsiveContainer width="100%" height={260}>
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
              </div>

              <div>
                <h4 className="mb-2 text-sm font-medium text-muted-foreground">Accuracy and mAP50 Comparison</h4>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={accuracyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 1]} label={{ value: "Score", angle: -90, position: "insideLeft" }} />
                    <Tooltip formatter={(value) => (typeof value === "number" ? `${(value * 100).toFixed(2)}%` : value)} />
                    <Legend />
                    <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
                    <Bar dataKey="map50" fill="#8b5cf6" name="mAP50" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {accelerationSummary.map((engine) => (
                <Card key={engine.engine} className="bg-muted/40">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">{engine.engine}</CardTitle>
                    <CardDescription>
                      {engine.available ? "Available" : "Unavailable"} | frames={engine.frames_evaluated}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="text-sm">
                    <div className="grid grid-cols-2 gap-2 text-muted-foreground">
                      <div>Avg Latency: {engine.avg_latency_ms !== null ? `${engine.avg_latency_ms.toFixed(2)}ms` : "N/A"}</div>
                      <div>Avg FPS: {engine.avg_fps !== null ? engine.avg_fps.toFixed(2) : "N/A"}</div>
                      <div>
                        Accuracy: {engine.evaluation?.accuracy !== null && engine.evaluation?.accuracy !== undefined
                          ? `${(engine.evaluation.accuracy * 100).toFixed(0)}%`
                          : "N/A"}
                      </div>
                      <div>
                        mAP50: {engine.evaluation?.map50 !== null && engine.evaluation?.map50 !== undefined
                          ? `${(engine.evaluation.map50 * 100).toFixed(0)}%`
                          : "N/A"}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Frame-by-Frame Detection Results</CardTitle>
          <CardDescription>
            Showing sampled frames with bounding boxes, per-frame latency, and detection count.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {results.frames.length === 0 ? (
            <p className="text-sm text-muted-foreground">No frames processed.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {results.frames.map((frame) => (
                <Card key={frame.frame_index} className="bg-muted/40">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Frame {frame.frame_index}</CardTitle>
                    <CardDescription>
                      t={frame.timestamp_sec.toFixed(2)}s | latency={frame.latency_ms.toFixed(2)}ms | detections={frame.num_detections}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      src={`data:image/jpeg;base64,${frame.preview_image_base64}`}
                      alt={`frame-${frame.frame_index}`}
                      className="w-full rounded-md border"
                    />
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                      <div>Accuracy: {frame.evaluation?.accuracy !== null && frame.evaluation?.accuracy !== undefined ? `${(frame.evaluation.accuracy * 100).toFixed(0)}%` : "N/A"}</div>
                      <div>mAP50: {frame.evaluation?.map50 !== null && frame.evaluation?.map50 !== undefined ? `${(frame.evaluation.map50 * 100).toFixed(0)}%` : "N/A"}</div>
                      <div>Precision: {frame.evaluation?.precision !== null && frame.evaluation?.precision !== undefined ? `${(frame.evaluation.precision * 100).toFixed(0)}%` : "N/A"}</div>
                      <div>Recall: {frame.evaluation?.recall !== null && frame.evaluation?.recall !== undefined ? `${(frame.evaluation.recall * 100).toFixed(0)}%` : "N/A"}</div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
