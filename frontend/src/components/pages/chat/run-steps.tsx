import { useLayoutEffect, useRef, useState } from "react"
import { ThinkingOrb } from "thinking-orbs"
import type { ChatStep } from "@/api/types"
import {
	formatDuration,
	orbState,
	runningStep,
	stepLabel,
} from "@/lib/chat-steps"

function StepIcon({ step, isRunning }: { step: ChatStep; isRunning: boolean }) {
	if (step.status === "completed") {
		return (
			<svg
				width="14"
				height="14"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				strokeWidth="2.5"
				strokeLinecap="round"
				strokeLinejoin="round"
				className="shrink-0 text-muted-foreground"
				aria-hidden
			>
				<path d="M20 6L9 17l-5-5" />
			</svg>
		)
	}
	if (!isRunning) {
		return (
			<span
				aria-hidden
				className="size-3 shrink-0 rounded-full border-[1.5px] border-border"
			/>
		)
	}
	return (
		<span
			aria-hidden
			className="size-3 shrink-0 animate-spin rounded-full border-[1.5px] border-border border-t-muted-foreground"
		/>
	)
}

/** The path a run took, above the answer it produced: the step it is on while it runs, and a
 * record of how the answer was reached once it settles. */
export function RunSteps({
	steps,
	isRunning,
}: {
	steps: ChatStep[]
	isRunning: boolean
}) {
	const [openedByReader, setOpenedByReader] = useState<boolean | null>(null)
	const traceRef = useRef<HTMLOListElement>(null)
	const [lineHeight, setLineHeight] = useState(0)
	const running = runningStep(steps)
	const isOpen = openedByReader ?? isRunning

	useLayoutEffect(() => {
		const trace = traceRef.current
		if (trace === null) return
		const observer = new ResizeObserver(() => setLineHeight(trace.offsetHeight))
		observer.observe(trace)
		return () => observer.disconnect()
	}, [])

	if (steps.length === 0 && !isRunning) return null

	const settled = `${steps.length} ${steps.length === 1 ? "step" : "steps"} · ${formatDuration(steps)}`

	return (
		<div className="flex w-full flex-col">
			<button
				type="button"
				aria-expanded={isOpen}
				onClick={() => setOpenedByReader(!isOpen)}
				className="-mx-1.5 flex w-fit items-center gap-2 rounded-md px-1.5 py-1 transition-colors hover:bg-muted"
			>
				<ThinkingOrb state={orbState(running)} size={20} paused={!isRunning} />
				<span role="status" className="contents">
					{isRunning ? (
						<span
							className="bg-clip-text font-medium text-[13px] text-transparent whitespace-nowrap"
							style={{
								backgroundImage:
									"linear-gradient(90deg, var(--muted-foreground) 35%, var(--foreground) 50%, var(--muted-foreground) 65%)",
								backgroundSize: "200% 100%",
								animation: "shimmer-text 1.4s linear infinite",
							}}
						>
							Working
						</span>
					) : (
						<span
							className="font-medium text-[13px] text-muted-foreground whitespace-nowrap"
							style={{ animation: "fade-in 350ms ease-out both" }}
						>
							{settled}
						</span>
					)}
				</span>
				<svg
					width="14"
					height="14"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					strokeWidth="2.2"
					strokeLinecap="round"
					strokeLinejoin="round"
					className="text-muted-foreground transition-transform duration-300"
					style={{ transform: isOpen ? "rotate(180deg)" : "rotate(0)" }}
					aria-hidden
				>
					<path d="M6 9l6 6 6-6" />
				</svg>
			</button>

			<div
				className="grid transition-[grid-template-rows,opacity] duration-400"
				style={{
					gridTemplateRows: isOpen ? "1fr" : "0fr",
					opacity: isOpen ? 1 : 0,
					transitionTimingFunction: "cubic-bezier(0.23, 1, 0.32, 1)",
				}}
			>
				<div className="overflow-hidden">
					<div className="relative mt-1 ml-[5px] pl-4">
						<span
							aria-hidden
							className="absolute left-[3px] w-px bg-border"
							style={{
								top: -8,
								height: lineHeight ? lineHeight - 2 : 0,
								transition: "height 500ms cubic-bezier(0.23,1,0.32,1)",
							}}
						/>
						<ol ref={traceRef} className="flex flex-col gap-1 py-1">
							{steps.map((step, index) => (
								<li
									// biome-ignore lint/suspicious/noArrayIndexKey: the trail only grows at its end, so a step's position is its identity
									key={index}
									className="flex min-h-7 w-full items-center gap-2 px-1.5 py-0.5"
									style={{
										animation: `fade-up 320ms cubic-bezier(0.23,1,0.32,1) both`,
									}}
								>
									<StepIcon step={step} isRunning={isRunning} />
									<span className="min-w-0 truncate font-medium text-[12.5px]">
										{stepLabel(step)}
									</span>
									{step.subject && (
										<span className="min-w-0 truncate text-[11.5px] text-muted-foreground">
											{step.subject}
										</span>
									)}
								</li>
							))}
						</ol>
					</div>
				</div>
			</div>
		</div>
	)
}
