// On-screen text boxes via Apple's Vision framework (no downloads, runs on the Neural Engine).
// Usage: vision_text <image> [<image> ...]
// Prints one line per box: <image index> <x0> <y0> <x1> <y1> <confidence> <text>
// in pixel coordinates with the origin at the top left.
import Foundation
import Vision
import AppKit

for (i, path) in CommandLine.arguments.dropFirst().enumerated() {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let w = Double(cg.width), h = Double(cg.height)
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    req.minimumTextHeight = 0.008
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    for obs in req.results ?? [] {
        guard let top = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox  // normalised, origin bottom-left
        let x0 = Int(b.minX * w), x1 = Int(b.maxX * w)
        let y0 = Int((1 - b.maxY) * h), y1 = Int((1 - b.minY) * h)
        let text = top.string.replacingOccurrences(of: "\n", with: " ")
        print("\(i) \(x0) \(y0) \(x1) \(y1) \(String(format: "%.2f", top.confidence)) \(text)")
    }
}
