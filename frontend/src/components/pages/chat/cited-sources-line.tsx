import type { ChatSource } from "@/api/types"
import { extractCitedMarkers } from "@/lib/citations"

export function CitedSourcesLine({
	answer,
	sources,
}: {
	answer: string
	sources: ChatSource[]
}) {
	const known = new Set(sources.map((source) => source.marker))
	const cited = extractCitedMarkers(answer, known)
	if (cited.length === 0) return null
	const noun = sources.length === 1 ? "passage" : "passages"
	return (
		<p className="text-muted-foreground text-xs">
			Cited {cited.length} of {sources.length} retrieved {noun}
		</p>
	)
}
