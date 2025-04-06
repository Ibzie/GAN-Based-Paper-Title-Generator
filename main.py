import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import random
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
import time

# Set seeds for reproducibility
seed = 99
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download NLTK resources if needed - these help with text processing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ArxivScraper:
    """
    Ethical Data Collection: This scraper respects ArXiv's servers with rate limiting
    and minimal data collection - only taking what we need for our GAN training.
    """
    
    def __init__(self, query="GANs", max_results=10):  # Reduced to 10 for quick demo - Can increase for better results
        self.query = query
        self.max_results = max_results
        self.base_url = "https://arxiv.org/search/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 Academic Research Bot (your-email-here@example.com)', # Replace with your email for good practice
        }
    
    def scrape_titles(self):
        """
        Gently scrapes paper titles from ArXiv with thoughtful rate limiting.
        Just as GANs balance two networks, we balance our need for data with server respect.
        """
        all_titles = []
        
        # Calculate the number of pages needed
        results_per_page = 25  # ArXiv typically shows 25 results per page
        num_pages = min(1, (self.max_results + results_per_page - 1) // results_per_page)
        
        print(f"Scraping up to {self.max_results} paper titles related to '{self.query}'")
        
        for page in range(num_pages):
            # Construct URL for the current page
            params = {
                'query': self.query,
                'searchtype': 'all',
                'source': 'header',
                'start': page * results_per_page
            }
            
            try:
                # Make request with rate limiting - being a good digital citizen
                response = requests.get(
                    self.base_url, 
                    params=params, 
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract paper titles
                    paper_elements = soup.select('p.title')
                    
                    for element in paper_elements:
                        title = element.text.strip()
                        all_titles.append(title)
                        
                        # Check if we've reached max_results
                        if len(all_titles) >= self.max_results:
                            break
                else:
                    print(f"Error: Received status code {response.status_code}")
                
                # Be ethical - don't hammer the server (like Gene shouldn't overfit to one story)
                time.sleep(1)  # Reduced to 1 second for the demo but still being respectful
                
            except Exception as e:
                print(f"Error during scraping: {e}")
            
            # Check if we've reached max_results
            if len(all_titles) >= self.max_results:
                break
        
        # If we couldn't scrape any titles (e.g., due to network issues), fall back to samples
        if not all_titles:
            print("Couldn't connect to ArXiv. Using sample titles instead.")
            all_titles = [
                "Generative Adversarial Networks for Image-to-Image Translation",
                "Progressive Growing of GANs for Improved Quality, Stability, and Variation",
                "Unsupervised Representation Learning with Deep Convolutional GANs",
                "Conditional Generative Adversarial Nets",
                "Wasserstein GAN",
                "Improved Techniques for Training GANs",
                "CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
                "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation",
                "Self-Attention Generative Adversarial Networks",
                "A Style-Based Generator Architecture for Generative Adversarial Networks"
            ][:self.max_results]
        
        print(f"Successfully scraped {len(all_titles)} paper titles")
        return all_titles
    
    def save_to_csv(self, titles, filename="arxiv_gan_papers.csv"):
        """Save the sample titles to a CSV file - our training data for the GAN"""
        df = pd.DataFrame({'title': titles})
        df.to_csv(filename, index=False)
        print(f"Saved {len(titles)} titles to {filename}")
        return df


# Text preprocessing: Our "latent space" preparation for academic paper titles
def preprocess_title(title, keep_stopwords=True):
    """
    Specialized preprocessing for academic paper titles - transforms raw text into
    a format suitable for our GAN, much like how the generator transforms random noise
    into structured data.
    """
    # Remove any special characters but keep important punctuation
    title = re.sub(r'[^\w\s:,\-()]', '', title)
    
    # Lowercase everything
    title = title.lower()
    
    # Tokenize
    tokens = word_tokenize(title)
    
    # Optionally filter stopwords
    if not keep_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Join back into a string
    processed_title = ' '.join(tokens)
    
    return processed_title


# Custom dataset class - the pipeline feeding data to our GAN
class TitleDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


# Meet Gene: The Generator - our creative storyteller from the article
class TitleGenerator(nn.Module):
    """
    This is our "Gene" - the Generator from our bar story analogy.
    It creates convincing paper titles from random noise, improving with each round of feedback.
    """
    def __init__(self, latent_dim, hidden_dim, vocab_size, seq_len, num_layers=1):  # Simplified to 1 layer
        super(TitleGenerator, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Process latent vector - this transforms random noise into the beginnings of a story
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, seq_len * hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Generate sequence with LSTM - crafting the narrative flow
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer - turning abstract thoughts into concrete words
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, z):
        # Process latent vector
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, self.seq_len, self.hidden_dim // 2)
        
        # Generate sequence
        lstm_output, _ = self.lstm(x)
        
        # Generate word logits
        logits = self.output(lstm_output)
        
        return logits


# Meet Dee: The Discriminator - our skeptical listener from the article
class TitleDiscriminator(nn.Module):
    """
    This is our "Dee" - the Discriminator who evaluates whether titles are real or fake.
    Like the friend in our analogy, it gets better at spotting fabrications with practice.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):  # Simplified to 1 layer
        super(TitleDiscriminator, self).__init__()
        
        # Embedding layer - understanding the meaning of each word
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequence processing - analyzing the flow and coherence of the story
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer - making the final "real or fake" judgment
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Embed sequence - translate words to meaningful vectors
        x = self.embedding(x)
        
        # Process sequence - analyze the narrative
        output, _ = self.lstm(x)
        
        # Global max pooling
        pooled, _ = torch.max(output, dim=1)
        
        # Generate validity prediction - the final verdict
        validity = self.fc(pooled)
        
        return validity


# Data preparation - creating the playground for Gene and Dee to compete
def prepare_title_data(titles, max_vocab_size=1000, max_seq_len=15):  # Simplified vocab and seq length
    """
    Transforms our raw titles into structured data for GAN training,
    building a vocabulary that defines our latent space dimensions.
    """
    print(f"Preparing data from {len(titles)} paper titles")
    
    # Preprocess titles
    print("Preprocessing titles...")
    processed_titles = [preprocess_title(title) for title in titles]
    
    # Split data
    train_titles, test_titles = train_test_split(processed_titles, test_size=0.1, random_state=seed)
    
    # Build vocabulary - the linguistic building blocks available to our Generator
    words = []
    for title in train_titles:
        words.extend(title.split())
    
    # Get most common words
    word_counts = Counter(words)
    most_common = word_counts.most_common(max_vocab_size - 4)  # Reserve 4 for special tokens
    
    # Create vocabulary mapping - the dictionary our networks use to communicate
    word_to_idx = {
        '<PAD>': 0,
        '<START>': 1,
        '<UNK>': 2,
        '<END>': 3
    }
    
    for word, _ in most_common:
        word_to_idx[word] = len(word_to_idx)
    
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Convert text to sequences - transforming human language to machine-readable format
    def title_to_sequence(title):
        words = title.split()
        seq = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
        seq = [word_to_idx['<START>']] + seq + [word_to_idx['<END>']]
        
        # Pad or truncate
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
        else:
            seq = seq + [word_to_idx['<PAD>']] * (max_seq_len - len(seq))
        
        return seq
    
    # Convert all texts
    train_sequences = [title_to_sequence(title) for title in train_titles]
    test_sequences = [title_to_sequence(title) for title in test_titles]
    
    return train_sequences, test_sequences, word_to_idx, idx_to_word, vocab_size, train_titles, test_titles


# Translator function - turning machine data back into human language
def indices_to_title(indices, idx_to_word):
    """
    Translates the Generator's numerical output back into readable text,
    letting us see the stories Gene is telling.
    """
    words = []
    for idx in indices:
        # Skip special tokens
        if idx in [0, 1]:  # PAD or START
            continue
        if idx == 3:  # END
            break
        words.append(idx_to_word.get(idx, '<UNK>'))
    return ' '.join(words)


# Temperature sampling - adding creativity to our Generator's stories
def sample_with_temperature(logits, temperature=1.0):
    """
    This is like Gene deciding how wild to make his stories.
    
    Low temperature (closer to 0): Gene tells conservative, predictable stories
    High temperature (closer to 1): Gene gets creative and takes more risks
    """
    if temperature == 0.0:
        # Greedy sampling - always choose the most likely word
        return torch.argmax(logits, dim=-1)
    
    # Apply temperature - adjust the creativity level
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = torch.softmax(scaled_logits, dim=-1)
    
    # Sample from the distribution - make choices with weighted randomness
    sampled_indices = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
    return sampled_indices.reshape(probs.size(0), probs.size(1))


# Quick demo training function - for educational purposes
def quick_train_gan(generator, discriminator, dataloader, latent_dim, device, idx_to_word):
    """
    A simplified training loop to demonstrate GAN concepts in seconds.
    This isn't meant for production use but for educational illustration.
    """
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.01)  # Higher learning rate for quick results
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.01)
    
    # Loss function
    criterion = nn.BCELoss()
    
    print("Demonstrating GAN training (quick mode)...")
    
    # Just a few iterations for demonstration
    for i in range(5):
        print(f"Training iteration {i+1}/5")
        
        # Get a batch of real data
        for real_sequences in dataloader:
            batch_size = real_sequences.size(0)
            real_sequences = real_sequences.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ---------------------
            #  Train Discriminator on real data
            # ---------------------
            d_optimizer.zero_grad()
            real_validity = discriminator(real_sequences)
            d_real_loss = criterion(real_validity, real_labels)
            
            # ---------------------
            #  Train Discriminator on fake data
            # ---------------------
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_logits = generator(z)
            fake_sequences = sample_with_temperature(fake_logits, 0.8).detach()
            fake_validity = discriminator(fake_sequences)
            d_fake_loss = criterion(fake_validity, fake_labels)
            
            # Combined discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            #  Train Generator
            # ---------------------
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_logits = generator(z)
            fake_sequences = sample_with_temperature(fake_logits, 0.8)
            fake_validity = discriminator(fake_sequences)
            g_loss = criterion(fake_validity, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            print(f"  D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            
            # Only process one batch per iteration for speed
            break
    
    # Generate samples
    print("\nGenerating sample titles:")
    generator.eval()
    with torch.no_grad():
        z = torch.randn(4, latent_dim).to(device)
        fake_logits = generator(z)
        fake_indices = sample_with_temperature(fake_logits, 0.8).cpu().numpy()
        for i, seq in enumerate(fake_indices):
            title = indices_to_title(seq, idx_to_word)
            print(f"  Sample {i+1}: {title}")
    
    return generator, discriminator


# Generate a batch of titles
def generate_titles(generator, latent_dim, device, idx_to_word, n=20, temperature=0.8):
    """Generate a batch of titles for demonstration"""
    generator.eval()
    generated_titles = []
    
    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        fake_logits = generator(z)
        fake_indices = sample_with_temperature(fake_logits, temperature).cpu().numpy()
        titles = [indices_to_title(seq, idx_to_word) for seq in fake_indices]
        generated_titles.extend(titles)
    
    # Filter out empty or very short titles
    generated_titles = [title for title in generated_titles if len(title.split()) >= 2]
    
    # Remove duplicates
    generated_titles = list(set(generated_titles))
    
    return generated_titles


def main():
    """
    Demonstration of a GAN for paper title generation.
    This is a simplified version running in about 10 seconds for educational purposes.
    """
    print("=== GAN Paper Title Generator ===")
    print("This demonstration shows how GANs can generate text by having two networks compete.")
    print("It runs in about 10 seconds to showcase the concept rather than achieve state-of-the-art results.\n")
    
    # Settings - simplified for quick demo
    latent_dim = 50  # Smaller latent dimension for speed
    hidden_dim = 64  # Smaller hidden dimension for speed
    embedding_dim = 32  # Smaller embedding dimension for speed
    max_seq_len = 15  # Shorter sequence length
    batch_size = 5  # Smaller batch size
    
    # 1. Get sample data
    print("\n1. Loading Sample Data")
    scraper = ArxivScraper(query="GANs", max_results=10)
    titles = scraper.scrape_titles()
    
    # 2. Process data
    print("\n2. Processing Data")
    train_sequences, test_sequences, word_to_idx, idx_to_word, vocab_size, train_titles, test_titles = prepare_title_data(
        titles, max_vocab_size=500, max_seq_len=max_seq_len
    )
    
    # 3. Create dataset and dataloader
    train_dataset = TitleDataset(train_sequences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Create models - simplified architecture for speed
    print("\n3. Creating GAN Models")
    generator = TitleGenerator(latent_dim, hidden_dim, vocab_size, max_seq_len, num_layers=1).to(device)
    discriminator = TitleDiscriminator(vocab_size, embedding_dim, hidden_dim, num_layers=1).to(device)
    
    # 5. Quick train for demonstration
    print("\n4. Demonstrating GAN Training (Quick Mode)")
    generator, discriminator = quick_train_gan(
        generator, discriminator, train_dataloader, latent_dim, device, idx_to_word
    )
    
    # 6. Generate synthetic titles
    print("\n5. Generating Synthetic Paper Titles")
    synthetic_titles = generate_titles(
        generator, latent_dim, device, idx_to_word, n=20, temperature=0.8
    )
    
    # 7. Save models
    print("\n6. Saving Models and Results")
    torch.save(generator.state_dict(), 'arxiv_title_generator.pth')
    torch.save(discriminator.state_dict(), 'arxiv_title_discriminator.pth')
    
    # 8. Save synthetic titles
    synthetic_df = pd.DataFrame({'title': synthetic_titles})
    synthetic_df.to_csv('synthetic_arxiv_titles.csv', index=False)
    
    print(f"\nGenerated {len(synthetic_titles)} synthetic paper titles:")
    for i, title in enumerate(synthetic_titles[:10]):  # Show only first 10
        print(f"  {i+1}. {title}")
    
    print("\nDemonstration complete! Check the generated files:")
    print("  - arxiv_title_generator.pth: The trained generator model")
    print("  - arxiv_title_discriminator.pth: The trained discriminator model")
    print("  - synthetic_arxiv_titles.csv: The generated synthetic titles")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")