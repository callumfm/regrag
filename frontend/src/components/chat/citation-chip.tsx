export function CitationChip({
	marker,
	onOpen,
}: {
	marker: number
	onOpen: (marker: number) => void
}) {
	return (
		<button
			type="button"
			aria-label={`Open source ${marker}`}
			onClick={() => onOpen(marker)}
			className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded-[4px] border border-primary/40 bg-primary/15 px-1 align-baseline font-medium text-[10px] text-primary tabular-nums hover:bg-primary/30"
		>
			{marker}
		</button>
	)
}
