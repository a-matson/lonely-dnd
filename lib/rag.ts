import { pipeline, FeatureExtractionPipeline } from "@huggingface/transformers";
import { openDB, IDBPDatabase } from "idb";
import MiniSearch from "minisearch";

// embeddings
let embedder: FeatureExtractionPipeline | null = null;

export async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      device: "webgpu",
    });
  }
  return embedder;
}

export async function getEmbedding(text: string): Promise<number[]> {
  const model = await getEmbedder();
  const output = await model(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// lightweight semantic chunking to preserve context
export function chunkText(text: string, maxTokens: number = 200, overlap: number = 50): string[] {
  const sentences = text.match(/[^.!?]+[.!?]+|\s*\n\s*/g) || [text];
  const chunks: string[] = [];
  let currentChunk = "";
  let currentTokens = 0;

  for (const sentence of sentences) {
    const sentenceTokens = Math.ceil(sentence.length / 4);
    if (currentTokens + sentenceTokens > maxTokens && currentChunk) {
      chunks.push(currentChunk.trim());
      currentChunk = currentChunk.slice(-Math.floor(overlap * 4)) + sentence;
      currentTokens = Math.ceil(currentChunk.length / 4);
    } else {
      currentChunk += " " + sentence;
      currentTokens += sentenceTokens;
    }
  }
  if (currentChunk.trim()) chunks.push(currentChunk.trim());
  return chunks;
}

// MiniSearch
interface RAGChunk {
  id: string;
  parentId: string;
  text: string;
  embedding: number[];
  timestamp: number;
}

let miniSearch = new MiniSearch<RAGChunk>({
  fields: ['text'],
  storeFields: ['id', 'text', 'parentId']
});

export async function initDB(): Promise<IDBPDatabase> {
  const db = await openDB("webllm-rag-db", 2, {
    upgrade(db) {
      if (!db.objectStoreNames.contains("chunks")) {
        db.createObjectStore("chunks", { keyPath: "id" });
      }
    },
  });

  // Hydrate MiniSearch on boot
  if (miniSearch.documentCount === 0) {
    const allChunks: RAGChunk[] = await db.getAll("chunks");
    if (allChunks.length > 0) {
      miniSearch.addAll(allChunks);
    }
  }
  return db;
}

export async function addDocument(text: string) {
  const db = await initDB();
  const parentId = crypto.randomUUID();
  const chunks = chunkText(text);

  for (let i = 0; i < chunks.length; i++) {
    const chunkTextStr = chunks[i];
    const embedding = await getEmbedding(chunkTextStr);
    
    const docChunk: RAGChunk = {
      id: `${parentId}-chunk-${i}`,
      parentId,
      text: chunkTextStr,
      embedding,
      timestamp: Date.now(),
    };
    
    await db.put("chunks", docChunk);
    
    // Add to keyword index
    if (!miniSearch.has(docChunk.id)) {
      miniSearch.add(docChunk);
    }
  }
}

// hybrid retrival & HyDE
function cosineSimilarity(a: number[], b: number[]) {
  let dotProduct = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// HyDE: generate a hypothetical answer to improve embedding matching
export async function generateHyDE(engine: any, query: string): Promise<string> {
  const prompt = `Please write a short, informative hypothetical paragraph that directly answers the following question. Do not include introductory filler. \n\nQuestion: ${query}`;
  
  const response = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    temperature: 0.3,
    max_tokens: 150
  });
  
  return response.choices[0].message.content;
}

// hybrid search: RRF of vector + keyword
export async function searchDocumentsHybrid(query: string, searchEmbedding: number[], topK: number = 3) {
  const db = await initDB();
  const allChunks: RAGChunk[] = await db.getAll("chunks");

  const vectorScores = allChunks.map(chunk => ({
    ...chunk,
    score: cosineSimilarity(searchEmbedding, chunk.embedding),
  })).sort((a, b) => b.score - a.score);

  const keywordResults = miniSearch.search(query, { fuzzy: 0.2 });
  
  const fusionScores: Record<string, { chunk: RAGChunk, score: number }> = {};
  
  const k = 60; // Standard RRF constant (https://www.ai21.com/glossary/tech/what-is-reciprocal-rank-fusion-rrf/)

  vectorScores.forEach((item, index) => {
    fusionScores[item.id] = { chunk: item, score: 1 / (k + index + 1) };
  });

  keywordResults.forEach((item, index) => {
    if (!fusionScores[item.id]) {
      const chunk = allChunks.find(c => c.id === item.id);
      if (chunk) fusionScores[item.id] = { chunk, score: 0 };
    }
    fusionScores[item.id].score += 1 / (k + index + 1);
  });

  const finalResults = Object.values(fusionScores)
    .sort((a, b) => b.score - a.score)
    .map(res => res.chunk)
    .slice(0, topK);

  return finalResults;
}