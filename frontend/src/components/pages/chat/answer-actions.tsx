import { Collapsible } from "@base-ui/react/collapsible"
import { CheckIcon, CopyIcon } from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import type { ChatSource } from "@/api/types"
import { citedSources, renumberCitations } from "@/lib/citations"
import { cn } from "@/lib/utils"
import { CITATION_BADGE } from "./citation-chip"

const COPIED_MS = 2000

function CopyAnswerButton({ text }: { text: string }) {
	const [copied, setCopied] = useState(false)
	const clearing = useRef<ReturnType<typeof setTimeout> | null>(null)

	useEffect(() => {
		return () => {
			if (clearing.current !== null) clearTimeout(clearing.current)
		}
	}, [])

	/** Silent where the clipboard is out of reach, since it needs a secure context. */
	async function copyAnswer() {
		try {
			await navigator.clipboard.writeText(text)
		} catch {
			return
		}
		setCopied(true)
		clearing.current = setTimeout(() => setCopied(false), COPIED_MS)
	}

	return (
		<button
			type="button"
			aria-label={copied ? "Answer copied" : "Copy answer"}
			onClick={copyAnswer}
			className="flex size-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
		>
			{copied ? <CheckIcon size={15} /> : <CopyIcon size={15} />}
		</button>
	)
}

/** What a reader can do with a settled answer, and the sources it stands on: the acts it cited,
 * numbered as the answer numbered them, each opening the passage it came from. */
export function AnswerActions({
	answer,
	sources,
	onOpenSource,
}: {
	answer: string
	sources: ChatSource[]
	onOpenSource: (marker: number) => void
}) {
	const cited = useMemo(() => citedSources(answer, sources), [answer, sources])
	const copyText = useMemo(
		() => renumberCitations(answer, sources),
		[answer, sources],
	)

	return (
		<Collapsible.Root className="fade-in animate-in mt-2 flex w-full flex-col duration-300">
			<div className="-mx-1 flex items-center gap-0.5">
				<CopyAnswerButton text={copyText} />
				{cited.length > 0 && (
					<Collapsible.Trigger className="ml-1 flex items-center gap-1.5 rounded-md px-1 py-0.5 transition-colors hover:bg-muted">
						<span className="flex -space-x-1">
							{cited.slice(0, 3).map(({ source, label }) => (
								<span
									key={source.marker}
									className={cn(CITATION_BADGE, "ring-2 ring-background")}
								>
									{label}
								</span>
							))}
						</span>
						<span className="text-muted-foreground text-xs">
							{cited.length} {cited.length === 1 ? "source" : "sources"}
						</span>
					</Collapsible.Trigger>
				)}
			</div>

			<Collapsible.Panel className="h-(--collapsible-panel-height) overflow-hidden transition-[height,opacity] duration-300 ease-[cubic-bezier(0.23,1,0.32,1)] data-ending-style:h-0 data-ending-style:opacity-0 data-starting-style:h-0 data-starting-style:opacity-0">
				<ul className="mt-1.5 flex flex-col rounded-lg border bg-muted p-1">
					{cited.map(({ source, label }) => (
						<li key={source.marker}>
							<button
								type="button"
								onClick={() => onOpenSource(source.marker)}
								className="flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left transition-colors hover:bg-background"
							>
								<span className={CITATION_BADGE}>{label}</span>
								<span className="truncate text-xs">{source.act}</span>
								<span className="ml-auto shrink-0 text-muted-foreground text-xs">
									{source.citation}
								</span>
							</button>
						</li>
					))}
				</ul>
			</Collapsible.Panel>
		</Collapsible.Root>
	)
}
