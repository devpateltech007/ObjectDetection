"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { VideoInferenceResponse } from "@/lib/api"

interface VideoInferenceResultsProps {
  results: VideoInferenceResponse
}

export function VideoInferenceResults({ results }: VideoInferenceResultsProps) {
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
