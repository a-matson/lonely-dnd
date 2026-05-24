import type { MLCEngineInterface } from "@mlc-ai/web-llm";
import { type IDBPDatabase, openDB } from "idb";
import type { Message } from "../app/page";
import { cosineSimilarity } from "../utils";
import { getEmbedding } from "./rag";

export interface MemoryChunk {
  id: string;
  text: string;
  embedding: number[];
  timestamp: number;
}

export interface NPCState {
  name: string;
  state: string;
  mood: string;
  current_action: string;
}

export interface GameState {
  id: string;
  location: string;
  time_and_weather: string;
  npcs: NPCState[];
}

// multi-tiered db
export async function initMemoryDB(): Promise<IDBPDatabase> {
  return await openDB("webllm-memory-db", 1, {
    upgrade(db) {
      if (!db.objectStoreNames.contains("episodic")) {
        db.createObjectStore("episodic", { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains("semantic")) {
        db.createObjectStore("semantic", { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains("gameState")) {
        db.createObjectStore("gameState", { keyPath: "id" });
      }
    },
  });
}

export async function getCurrentGameState(): Promise<GameState> {
  const db = await initMemoryDB();
  const state = await db.get("gameState", "current-state");

  if (!state) {
    return {
      id: "current-state",
      location: "Unknown Location",
      time_and_weather: "Clear",
      npcs: [],
    };
  }
  return state;
}

export async function saveGameState(newState: Omit<GameState, "id">) {
  const db = await initMemoryDB();
  await db.put("gameState", { id: "current-state", ...newState });
}

// episodic memory
export async function archiveToEpisodicMemory(messages: Message[]) {
  const db = await initMemoryDB();
  const textToArchive = messages
    .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
    .join("\n");
  const embedding = await getEmbedding(textToArchive);

  const memory: MemoryChunk = {
    id: crypto.randomUUID(),
    text: textToArchive,
    embedding,
    timestamp: Date.now(),
  };

  await db.put("episodic", memory);
}

// semantic memory
export async function extractAndStoreFacts(
  engine: MLCEngineInterface,
  userMessage: string,
  dmNarrative: string,
) {
  const db = await initMemoryDB();

  const prompt = `
    Analyze the following D&D turn (Player Action + DM Narrative).
    Extract any concrete facts, newly introduced items, environmental details, or NPC traits that were established in this turn.
    Do not extract basic combat mechanics (like "goblin took 5 damage"). Focus on permanent lore, inventory, or world-building facts.
    Format the output strictly as a JSON object with a single key "facts" containing an array of strings. 
    If there are no facts to extract, return {"facts": []}.
    
    Player Action: "${userMessage}"
    DM Narrative: "${dmNarrative}"
  `;

  const factSchema = {
    type: "object",
    properties: { facts: { type: "array", items: { type: "string" } } },
    required: ["facts"],
  };

  try {
    const response = await engine.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      temperature: 0.1,
      response_format: {
        type: "json_object",
        schema: JSON.stringify(factSchema),
      },
    });

    const resultStr = response.choices[0].message.content;
    const parsed = JSON.parse(resultStr || '{"facts": []}');

    if (
      parsed.facts &&
      Array.isArray(parsed.facts) &&
      parsed.facts.length > 0
    ) {
      const existingSemanticMemories: MemoryChunk[] =
        await db.getAll("semantic");

      for (const fact of parsed.facts) {
        const embedding = await getEmbedding(fact);

        let isDuplicate = false;
        for (const existingMem of existingSemanticMemories) {
          const similarity = cosineSimilarity(embedding, existingMem.embedding);
          if (similarity > 0.85) {
            isDuplicate = true;
            break;
          }
        }

        if (!isDuplicate) {
          const memory: MemoryChunk = {
            id: crypto.randomUUID(),
            text: fact,
            embedding,
            timestamp: Date.now(),
          };
          await db.put("semantic", memory);
          console.log("Extracted and saved new narrative fact:", fact);
          existingSemanticMemories.push(memory);
        }
      }
    }
  } catch (e) {
    console.warn("Fact extraction failed:", e);
  }
}

// query memory
export async function retrieveRelevantMemory(
  queryEmbedding: number[],
  topK: number = 3,
) {
  const db = await initMemoryDB();

  const episodic: MemoryChunk[] = await db.getAll("episodic");
  const semantic: MemoryChunk[] = await db.getAll("semantic");

  const now = Date.now();

  const episodicScores = episodic
    .map((chunk) => {
      const ageInHours = (now - chunk.timestamp) / (1000 * 60 * 60);

      // Time decay factor: newer memories stay close to a 1.0 multiplier.
      // Older memories slowly decay down to a minimum of 0.5.
      const timeDecay = Math.max(0.5, 1 - ageInHours * 0.05);

      const semanticScore = cosineSimilarity(queryEmbedding, chunk.embedding);

      return {
        ...chunk,
        type: "episodic",
        score: semanticScore * timeDecay,
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  // score semantic memories (core facts) - NO time decay
  const semanticScores = semantic
    .map((chunk) => ({
      ...chunk,
      type: "semantic",
      score: cosineSimilarity(queryEmbedding, chunk.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  return {
    episodic: episodicScores.filter((s) => s.score > 0.45),
    semantic: semanticScores.filter((s) => s.score > 0.55),
  };
}
