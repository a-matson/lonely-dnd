import { Txt2ImgWorkerClient } from 'web-txt2img';

// Keep a singleton instance of the client
let avatarGenerator: Txt2ImgWorkerClient | null = null;

export async function getAvatarGenerator() {
  if (!avatarGenerator) {
    avatarGenerator = Txt2ImgWorkerClient.createDefault();
    
    // Load a lightweight, fast diffusion model natively supported by the library
    // SD-Turbo takes about ~2GB of storage. It requires WebGPU to run reliably.
    await avatarGenerator.load('sd-turbo', { 
        backendPreference: ['webgpu'] 
    });
  }
  return avatarGenerator;
}

export async function generateLocalAvatar(description: string): Promise<string> {
  const generator = await getAvatarGenerator();
  
  // Prompt engineering for RPG avatars
  const prompt = `retro 8-bit pixel art portrait, ${description}, solid background, sharp pixels, fantasy RPG character portrait`;
  
  // Trigger generation
  const { promise } = generator.generate({ 
      prompt: prompt, 
      seed: Math.floor(Math.random() * 1000000),
  });

  const result = await promise;
  
  if (result.ok && result.blob) {
      // The result blob is 512x512. We need to downscale it to 64x64.
      return await downscaleTo64x64(result.blob);
  } else {
      console.error("Local avatar generation failed");
      return "";
  }
}

// Helper to downscale the image to exactly 64x64 pixels for that retro look
function downscaleTo64x64(imageBlob: Blob): Promise<string> {
  return new Promise((resolve) => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    
    if (!ctx) return resolve("");

    // Disable image smoothing to keep the pixelated look
    ctx.imageSmoothingEnabled = false;

    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, 64, 64);
      resolve(canvas.toDataURL("image/png"));
      URL.revokeObjectURL(img.src); // cleanup
    };
    
    img.src = URL.createObjectURL(imageBlob);
  });
}