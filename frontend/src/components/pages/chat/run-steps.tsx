import { ChevronRightIcon } from "lucide-react"
import { useState } from "react"
import type { ChatStep } from "@/api/types"
import { formatDuration, stepLabel, trailSteps } from "@/lib/chat-steps"
import { cn } from "@/lib/utils"

/** The path a run took, above the answer it produced: open while the run walks it, and
 * collapsed to a summary once it settles — a record of how the answer was reached, which
 * is what to read when the answer is wrong. */
export function RunSteps({
	steps,
	isRunning,
}: {
	steps: ChatStep[]
	isRunning: boolean
}) {
	const [openedByReader, setOpenedByReader] = useState<boolean | null>(null)
	const trail = trailSteps(steps)
	if (trail.length === 0 && !isRunning) return null

	const isOpen = openedByReader ?? isRunning

	return (
		<div className="text-xs">
			<button
				type="button"
				aria-expanded={isOpen}
				onClick={() => setOpenedByReader(!isOpen)}
				className="flex items-center gap-1 text-muted-foreground hover:text-foreground"
			>
				<ChevronRightIcon
					className={cn("size-3 transition-transform", isOpen && "rotate-90")}
				/>
				<span className={cn(isRunning && "animate-pulse")}>
					{isRunning
						? "Working…"
						: `${trail.length} ${trail.length === 1 ? "step" : "steps"} · ${formatDuration(trail)}`}
				</span>
			</button>
			{isOpen && trail.length > 0 && (
				<ol className="mt-1.5 ml-1.5 flex flex-col gap-1 border-border border-l pl-3 text-muted-foreground">
					{trail.map((step, index) => (
						// biome-ignore lint/suspicious/noArrayIndexKey: the trail only ever grows at its end, so a step's position is fixed once it arrives
						<li key={index} className="flex gap-2">
							<span className="text-foreground/70">{stepLabel(step)}</span>
							{step.subject && (
								<span className="min-w-0 truncate italic">{step.subject}</span>
							)}
						</li>
					))}
				</ol>
			)}
		</div>
	)
}
