import { useEffect, useState } from "react";
import {
  deleteMemory,
  getAllMemories,
  type MemoryChunk,
  updateMemory,
} from "../lib/memory";
import {
  deleteDocument,
  getAllDocuments,
  type RAGChunk,
  updateDocument,
} from "../lib/rag";

interface MemoryManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function MemoryManager({ isOpen, onClose }: MemoryManagerProps) {
  const [activeTab, setActiveTab] = useState<"facts" | "memories" | "rules">(
    "facts",
  );
  const [menuData, setMenuData] = useState<{
    facts: MemoryChunk[];
    memories: MemoryChunk[];
    rules: RAGChunk[];
  }>({ facts: [], memories: [], rules: [] });

  async function loadMenuData() {
    const [facts, memories, rules] = await Promise.all([
      getAllMemories("semantic"),
      getAllMemories("episodic"),
      getAllDocuments(),
    ]);
    setMenuData({ facts, memories, rules });
  }

  useEffect(() => {
    if (isOpen) {
      loadMenuData();
    }
  }, [isOpen]);

  async function handleEditItem(
    id: string,
    currentText: string,
    type: "semantic" | "episodic" | "rule",
  ) {
    const newText = window.prompt("Edit entry:", currentText);
    if (!newText || newText === currentText) return;

    if (type === "rule") {
      await updateDocument(id, newText);
    } else {
      await updateMemory(type, id, newText);
    }
    await loadMenuData();
  }

  async function handleDeleteItem(
    id: string,
    type: "semantic" | "episodic" | "rule",
  ) {
    if (!window.confirm("Delete this entry forever?")) return;

    if (type === "rule") {
      await deleteDocument(id);
    } else {
      await deleteMemory(type, id);
    }
    await loadMenuData();
  }

  const renderList = (items: any[], type: "semantic" | "episodic" | "rule") => {
    if (items.length === 0)
      return <p className="text-gray-500 text-sm">Nothing stored here yet.</p>;
    return items.map((item) => (
      <div
        key={item.id}
        className="p-3 bg-gray-800 rounded border border-gray-700 flex justify-between gap-4 items-start mb-2 shadow"
      >
        <p className="text-sm flex-1 whitespace-pre-wrap text-gray-200">
          {item.text}
        </p>
        <div className="flex gap-2 shrink-0">
          <button
            type="button"
            onClick={() => handleEditItem(item.id, item.text, type)}
            className="text-xs bg-blue-900/50 hover:bg-blue-800 text-blue-200 px-2 py-1 rounded"
          >
            Edit
          </button>
          <button
            type="button"
            onClick={() => handleDeleteItem(item.id, type)}
            className="text-xs bg-red-900/50 hover:bg-red-800 text-red-200 px-2 py-1 rounded"
          >
            Delete
          </button>
        </div>
      </div>
    ));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4 backdrop-blur-sm">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-3xl h-[80vh] flex flex-col shadow-2xl">
        <div className="p-4 border-b border-gray-800 flex justify-between items-center bg-gray-950 rounded-t-lg">
          <h2 className="text-xl font-bold text-amber-500">Database Manager</h2>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-white font-bold text-xl px-2"
          >
            &times;
          </button>
        </div>

        <div className="flex border-b border-gray-800 bg-gray-900">
          <button
            type="button"
            onClick={() => setActiveTab("facts")}
            className={`flex-1 py-3 text-sm font-semibold transition-colors ${activeTab === "facts" ? "border-b-2 border-amber-500 text-amber-500" : "text-gray-400 hover:text-gray-200"}`}
          >
            Character Facts (Semantic)
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("memories")}
            className={`flex-1 py-3 text-sm font-semibold transition-colors ${activeTab === "memories" ? "border-b-2 border-amber-500 text-amber-500" : "text-gray-400 hover:text-gray-200"}`}
          >
            Past Events (Episodic)
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("rules")}
            className={`flex-1 py-3 text-sm font-semibold transition-colors ${activeTab === "rules" ? "border-b-2 border-amber-500 text-amber-500" : "text-gray-400 hover:text-gray-200"}`}
          >
            Campaign Lore (RAG)
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 bg-gray-900/50">
          {activeTab === "facts" && renderList(menuData.facts, "semantic")}
          {activeTab === "memories" &&
            renderList(menuData.memories, "episodic")}
          {activeTab === "rules" && renderList(menuData.rules, "rule")}
        </div>
      </div>
    </div>
  );
}
