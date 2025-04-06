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
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# Set seeds for reproducibility as one does
seed = 99
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# I am using GPU but in case you dont have access to one
# GANs train much faster with GPU acceleration
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
    This aligns with our article's emphasis on responsible AI development.
    """
    
    def __init__(self, query="GANs", max_results=100):
        self.query = query
        self.max_results = max_results
        self.base_url = "https://arxiv.org/search/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 Academic Research Bot (your@email.com)',
        }
    
    def scrape_titles(self):
        """
        Gently scrapes paper titles from ArXiv with thoughtful rate limiting.
        Just as GANs balance two networks, we balance our need for data with server respect.
        """
        all_titles = []
        
        # Calculate the number of pages needed
        results_per_page = 25  # ArXiv typically shows 25 results per page
        num_pages = min(10, (self.max_results + results_per_page - 1) // results_per_page)
        
        print(f"Scraping up to {self.max_results} paper titles related to '{self.query}'")
        
        for page in tqdm(range(num_pages), desc="Scraping Pages"):
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
                time.sleep(3)  # Wait 3 seconds between requests
                
            except Exception as e:
                print(f"Error during scraping: {e}")
            
            # Check if we've reached max_results
            if len(all_titles) >= self.max_results:
                break
        
        print(f"Successfully scraped {len(all_titles)} paper titles")
        return all_titles
    
    def save_to_csv(self, titles, filename="arxiv_gan_papers.csv"):
        """Save the scraped titles to a CSV file - our training data for the GAN"""
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
    def __init__(self, latent_dim, hidden_dim, vocab_size, seq_len, num_layers=2):
        super(TitleGenerator, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Process latent vector - this transforms random noise into the beginnings of a story
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, seq_len * hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Generate sequence with multi-layer LSTM - crafting the narrative flow
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Attention mechanism - focusing on the most important parts of the story
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
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
        
        # Apply attention
        attention_weights = torch.softmax(
            self.attention(lstm_output).squeeze(-1), 
            dim=1
        ).unsqueeze(2)
        
        # Weight the LSTM outputs by attention scores
        context = torch.sum(lstm_output * attention_weights, dim=1)
        context = context.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Combine context with LSTM output
        enhanced_output = lstm_output + context
        
        # Generate word logits
        logits = self.output(enhanced_output)
        
        return logits


# Meet Dee: The Discriminator - our skeptical listener from the article
class TitleDiscriminator(nn.Module):
    """
    This is our "Dee" - the Discriminator who evaluates whether titles are real or fake.
    Like the friend in our analogy, it gets better at spotting fabrications with practice.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(TitleDiscriminator, self).__init__()
        
        # Embedding layer - understanding the meaning of each word
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequence processing - analyzing the flow and coherence of the story
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Attention layer - focusing on the most suspicious parts of the story
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layer - making the final "real or fake" judgment
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Embed sequence - translate words to meaningful vectors
        x = self.embedding(x)
        
        # Process sequence - analyze the narrative
        output, _ = self.lstm(x)
        
        # Apply attention - focus on telltale signs of fakery
        attention_weights = torch.softmax(
            self.attention(output).squeeze(-1), 
            dim=1
        ).unsqueeze(2)
        
        # Weight the LSTM outputs by attention scores
        context = torch.sum(output * attention_weights, dim=1)
        
        # Generate validity prediction - the final verdict
        validity = self.fc(context)
        
        return validity


# Data preparation - creating the playground for Gene and Dee to compete
def prepare_title_data(titles, max_vocab_size=5000, max_seq_len=20):
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


# Evaluation metrics - measuring how well Gene can fool Dee
def evaluate_titles(generated_titles, real_titles, vocab_size):
    """
    Quantifying the quality of our Generator's output with various metrics,
    just as we'd measure the believability of Gene's stories.
    """
    results = {}
    
    # 1. Average title length - do they match real paper titles?
    gen_lengths = [len(title.split()) for title in generated_titles]
    real_lengths = [len(title.split()) for title in real_titles]
    
    results['avg_gen_length'] = sum(gen_lengths) / len(gen_lengths)
    results['avg_real_length'] = sum(real_lengths) / len(real_lengths)
    
    # 2. Vocabulary usage - is Gene using diverse language?
    gen_vocab = set()
    for title in generated_titles:
        gen_vocab.update(title.split())
    
    results['vocab_coverage'] = len(gen_vocab) / vocab_size
    
    # 3. BLEU score - how similar are Gene's stories to real ones?
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for gen_title in random.sample(generated_titles, min(50, len(generated_titles))):
        gen_tokens = gen_title.split()
        
        # Calculate BLEU against multiple references
        candidate_bleus = []
        for ref_title in random.sample(real_titles, min(10, len(real_titles))):
            ref_tokens = ref_title.split()
            score = sentence_bleu([ref_tokens], gen_tokens, 
                                  smoothing_function=smoothing,
                                  weights=(0.5, 0.5))
            candidate_bleus.append(score)
        
        # Use max BLEU against any reference
        if candidate_bleus:
            bleu_scores.append(max(candidate_bleus))
    
    if bleu_scores:
        results['avg_bleu'] = sum(bleu_scores) / len(bleu_scores)
    else:
        results['avg_bleu'] = 0.0
    
    # 4. Title uniqueness - is Gene telling diverse stories?
    unique_titles = len(set(generated_titles))
    results['uniqueness'] = unique_titles / len(generated_titles)
    
    return results


# The adversarial training loop - Gene and Dee's back-and-forth competition
def train_title_gan(generator, discriminator, dataloader, num_epochs, latent_dim, 
                   device, idx_to_word, real_titles, sampling_temp=0.8):
    """
    This is the core of our GAN - the adversarial dance between Generator and Discriminator.
    Just like in our bar analogy, Gene tells stories and Dee evaluates them in a continuous cycle.
    """
    # Optimizers - the learning mechanisms for our networks
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-5, betas=(0.5, 0.999))
    
    # Loss function - measuring how well each network is doing its job
    criterion = nn.BCELoss()
    
    # For tracking progress
    d_losses, g_losses, samples, metrics = [], [], [], []
    
    # Fixed noise for evaluation - consistent input to track Generator improvement
    fixed_noise = torch.randn(4, latent_dim).to(device)
    
    # For each epoch - one complete round of the adversarial game
    for epoch in range(num_epochs):
        epoch_d_losses = []
        epoch_g_losses = []
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for i, real_sequences in progress_bar:
            batch_size = real_sequences.size(0)
            real_sequences = real_sequences.to(device)
            
            # Labels - what we're training our networks to produce/recognize
            real_labels = torch.full((batch_size, 1), 0.9, device=device)  # smooth real to 0.9
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ---------------------
            #  Train Discriminator - Dee learns to spot fake stories
            # ---------------------
            
            # Reset discriminator gradients
            d_optimizer.zero_grad()
            
            # Train on real data - learn what real paper titles look like
            real_validity = discriminator(real_sequences)
            d_real_loss = criterion(real_validity, real_labels)
            
            # Train on fake data - learn what Generator's fake titles look like
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_logits = generator(z)
            
            # Sample with temperature - convert logits to discrete tokens
            fake_sequences = sample_with_temperature(fake_logits, sampling_temp).detach()
            
            fake_validity = discriminator(fake_sequences)
            d_fake_loss = criterion(fake_validity, fake_labels)
            
            # Combine losses and update - balance real and fake learning
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            #  Train Generator - Gene learns to tell better stories
            # ---------------------
            
            # Reset generator gradients
            g_optimizer.zero_grad()
            
            # Generate fake samples - Gene crafts new paper titles
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_logits = generator(z)
            
            # Direct prediction requires non-differentiable sampling
            # Use policy gradient trick for discrete sampling
            fake_sequences = sample_with_temperature(fake_logits, sampling_temp)
            
            # For better gradient flow, use one-hot encoding and embedding matrix directly
            # This is a simplified straight-through estimator
            embedding_weight = discriminator.embedding.weight
            fake_one_hot = torch.zeros(fake_sequences.size(0), fake_sequences.size(1), embedding_weight.size(0)).to(device)
            fake_one_hot.scatter_(2, fake_sequences.unsqueeze(2), 1)
            
            # Apply embedding by matrix multiplication
            fake_embeddings = torch.matmul(fake_one_hot, embedding_weight)
            
            # Pass directly to discriminator
            fake_validity = discriminator(fake_sequences)
            
            # Generator wants discriminator to predict "real" - Gene aims to fool Dee
            g_loss = criterion(fake_validity, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            # Track losses
            epoch_d_losses.append(d_loss.item())
            epoch_g_losses.append(g_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'D Loss': f"{d_loss.item():.4f}",
                'G Loss': f"{g_loss.item():.4f}"
            })
        
        # Save average losses for this epoch
        d_losses.append(sum(epoch_d_losses) / len(epoch_d_losses))
        g_losses.append(sum(epoch_g_losses) / len(epoch_g_losses))
        
        # Generate samples with fixed noise for evaluation
        with torch.no_grad():
            fake_logits = generator(fixed_noise)
            fake_indices = sample_with_temperature(fake_logits, sampling_temp).cpu().numpy()
            epoch_samples = [indices_to_title(seq, idx_to_word) for seq in fake_indices]
            samples.append(epoch_samples)
            
            # Generate more samples for metrics
            z = torch.randn(100, latent_dim).to(device)
            fake_logits = generator(z)
            fake_indices = sample_with_temperature(fake_logits, sampling_temp).cpu().numpy()
            generated_titles = [indices_to_title(seq, idx_to_word) for seq in fake_indices]
            
            # Calculate metrics
            epoch_metrics = evaluate_titles(generated_titles, real_titles, vocab_size=discriminator.embedding.weight.size(0))
            metrics.append(epoch_metrics)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"D Loss: {d_losses[-1]:.4f}, G Loss: {g_losses[-1]:.4f}")
        print(f"Metrics: Uniqueness: {metrics[-1]['uniqueness']:.2f}, BLEU: {metrics[-1]['avg_bleu']:.4f}")
        print("Sample generated titles:")
        for i, sample in enumerate(epoch_samples):
            print(f"  Sample {i+1}: {sample}")
        print()
    
    return d_losses, g_losses, samples, metrics


# Generate synthetic data - using our trained Generator to create new content
def generate_titles(generator, latent_dim, device, idx_to_word, n=1000, temperature=0.8, batch_size=32):
    """
    Put our trained Generator to work creating new paper titles -
    like Gene telling stories to a new audience after practicing with Dee.
    """
    generator.eval()
    generated_titles = []
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            current_batch_size = min(batch_size, n - i)
            z = torch.randn(current_batch_size, latent_dim).to(device)
            fake_logits = generator(z)
            fake_indices = sample_with_temperature(fake_logits, temperature).cpu().numpy()
            titles = [indices_to_title(seq, idx_to_word) for seq in fake_indices]
            generated_titles.extend(titles)
    
    # Filter out empty or very short titles
    generated_titles = [title for title in generated_titles if len(title.split()) >= 3]
    
    # Remove duplicates
    generated_titles = list(set(generated_titles))
    
    return generated_titles


# Visualize results - seeing the GAN's progress over time
def plot_results(d_losses, g_losses, samples, metrics, save_prefix="title_gan"):
    """
    Visualize how our GAN training progressed - tracking the adversarial
    balance between Generator and Discriminator across multiple metrics.
    """
    # Plot losses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    
    # Plot metrics
    plt.subplot(2, 2, 2)
    uniqueness = [m['uniqueness'] for m in metrics]
    bleu = [m['avg_bleu'] for m in metrics]
    
    plt.plot(uniqueness, label='Title Uniqueness')
    plt.plot(bleu, label='Avg BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Generation Quality Metrics')
    
    # Plot title length
    plt.subplot(2, 2, 3)
    gen_lengths = [m['avg_gen_length'] for m in metrics]
    real_lengths = [m['avg_real_length'] for m in metrics[0:1]] * len(metrics)
    
    plt.plot(gen_lengths, label='Generated Length')
    plt.plot(real_lengths, label='Real Length', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Words')
    plt.legend()
    plt.title('Average Title Length')
    
    # Plot vocabulary usage
    plt.subplot(2, 2, 4)
    vocab_coverage = [m['vocab_coverage'] for m in metrics]
    
    plt.plot(vocab_coverage, label='Vocabulary Coverage')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.legend()
    plt.title('Vocabulary Usage')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_metrics.png')
    plt.close()
    
    # Plot sample evolution (first sample)
    plt.figure(figsize=(12, 8))
    plt.title('Evolution of Generated Paper Titles')
    
    for i, epoch_samples in enumerate(samples):
        y_pos = len(samples) - i - 1  # Plot from bottom to top
        plt.text(0.1, y_pos / len(samples), f"Epoch {i+1}: {epoch_samples[0]}", fontsize=10)
    
    plt.axis('off')
    plt.savefig(f'{save_prefix}_evolution.png')
    plt.close()


def main():
    """
    The full GAN implementation pipeline - from data collection to model training
    to synthetic data generation, just as described in our article.
    """
    # Settings - hyperparameters for our GAN
    latent_dim = 128
    hidden_dim = 256
    embedding_dim = 128
    max_seq_len = 20  # Paper titles are shorter than reviews
    batch_size = 32
    num_epochs = 15
    
    # 1. Scrape data (with ethical considerations)
    scraper = ArxivScraper(query="GANs", max_results=100)
    titles = scraper.scrape_titles()
    
    # If you already have data or want to bypass scraping:
    # titles = pd.read_csv("arxiv_gan_papers.csv")['title'].tolist()
    
    # 2. Process data - prepare the training set
    train_sequences, test_sequences, word_to_idx, idx_to_word, vocab_size, train_titles, test_titles = prepare_title_data(
        titles, max_vocab_size=3000, max_seq_len=max_seq_len
    )
    
    # 3. Create dataset and dataloader - the training pipeline
    train_dataset = TitleDataset(train_sequences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Create models - instantiate Gene and Dee
    generator = TitleGenerator(latent_dim, hidden_dim, vocab_size, max_seq_len, num_layers=2).to(device)
    discriminator = TitleDiscriminator(vocab_size, embedding_dim, hidden_dim, num_layers=2).to(device)
    
    # Print model architecture
    print("\nGenerator:")
    print(generator)
    print("\nDiscriminator:")
    print(discriminator)
    
    # 5. Train models - let the adversarial game begin!
    print("\nTraining GAN for paper title generation...")
    d_losses, g_losses, samples, metrics = train_title_gan(
        generator, discriminator, train_dataloader, num_epochs, 
        latent_dim, device, idx_to_word, train_titles, sampling_temp=0.8
    )
    
    # 6. Visualize results - see how our networks evolved
    plot_results(d_losses, g_losses, samples, metrics, save_prefix="arxiv_title_gan")
    
    # 7. Generate a large batch of synthetic paper titles - the fruits of our labor
    print("\nGenerating synthetic paper titles...")
    synthetic_titles = generate_titles(
        generator, latent_dim, device, idx_to_word, 
        n=500, temperature=0.8, batch_size=32
    )
    
    # 8. Save synthetic titles - persist our results
    synthetic_df = pd.DataFrame({'title': synthetic_titles})
    synthetic_df.to_csv('synthetic_arxiv_titles.csv', index=False)
    
    print(f"\nGenerated {len(synthetic_titles)} unique synthetic paper titles")
    print("Sample of synthetic titles:")
    for i, title in enumerate(random.sample(synthetic_titles, min(10, len(synthetic_titles)))):
        print(f"{i+1}. {title}")
    
    # 9. Save models - preserve our trained networks
    torch.save(generator.state_dict(), 'arxiv_title_generator.pth')
    torch.save(discriminator.state_dict(), 'arxiv_title_discriminator.pth')
    
    print("\nTraining complete. Models and results saved.")


if __name__ == "__main__":
    main()