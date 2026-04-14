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
