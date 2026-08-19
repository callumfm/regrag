import { useRef } from "react"
import type { ChatSource } from "@/api/types"
import {
	Drawer,
	DrawerContent,
	DrawerDescription,
	DrawerHeader,
	DrawerTitle,
} from "@/components/ui/drawer"
import { CITATION_BADGE } from "./citation-chip"

export function SourcePanel({
	source,
	label,
	onClose,
}: {
	source: ChatSource | null
	label: number | null
	onClose: () => void
}) {
	const lastOpened = useRef<{ source: ChatSource; label: number | null }>(null)
	if (source !== null) lastOpened.current = { source, label }
	const shown = lastOpened.current

	return (
		<Drawer
			open={source !== null}
			onOpenChange={(open: boolean) => {
				if (!open) onClose()
			}}
			swipeDirection="right"
			modal={false}
		>
			<DrawerContent>
				<DrawerHeader>
					<DrawerTitle className="flex items-center gap-2">
						{shown?.label != null && (
							<span className={CITATION_BADGE}>{shown.label}</span>
						)}
						{shown?.source.citation}
					</DrawerTitle>
					<DrawerDescription>
						{shown?.source.celex}
						{shown?.source.title ? ` · ${shown.source.title}` : ""}
					</DrawerDescription>
				</DrawerHeader>
				<div className="flex-1 overflow-y-auto p-4 text-sm leading-relaxed whitespace-pre-wrap">
					{shown?.source.text}
				</div>
			</DrawerContent>
		</Drawer>
	)
}
