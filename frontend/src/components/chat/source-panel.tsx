import { CITATION_BADGE } from '@/components/chat/citation-chip'
import {
	Drawer,
	DrawerContent,
	DrawerDescription,
	DrawerHeader,
	DrawerTitle,
} from '@/components/ui/drawer'
import type { ChatSource } from '@/lib/chat/use-chat-stream'

export function SourcePanel({
	source,
	label,
	onClose,
}: {
	source: ChatSource | null
	label: number | null
	onClose: () => void
}) {
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
						{label !== null && <span className={CITATION_BADGE}>{label}</span>}
						{source?.citation}
					</DrawerTitle>
					<DrawerDescription>
						{source?.celex}
						{source?.title ? ` · ${source.title}` : ''}
					</DrawerDescription>
				</DrawerHeader>
				<div className="flex-1 overflow-y-auto p-4 text-sm leading-relaxed whitespace-pre-wrap">
					{source?.text}
				</div>
			</DrawerContent>
		</Drawer>
	)
}
