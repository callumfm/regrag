export const CITATION_BADGE =
	'inline-flex h-4 min-w-4 items-center justify-center rounded-[4px] border border-primary/70 bg-primary/30 px-1 font-medium text-[10px] text-[oklch(from_var(--primary)_0.62_calc(c*1.6)_h)] tabular-nums dark:text-primary'

export function CitationChip({
	marker,
	label,
	onOpen,
}: {
	marker: number
	label: number
	onOpen: (marker: number) => void
}) {
	return (
		<sup className="ml-px text-[0.7em] leading-none">
			<button
				type="button"
				aria-label={`Open source ${label}`}
				onClick={() => onOpen(marker)}
				className="rounded-sm px-0.5 font-medium text-[oklch(from_var(--primary)_0.62_calc(c*1.6)_h)] tabular-nums hover:bg-primary/20 hover:underline dark:text-primary"
			>
				[{label}]
			</button>
		</sup>
	)
}
