import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout, LeakyReLU, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION CONSTANTS ---
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 128, 128, 3
LATENT_DIM = 100  # Dimension of the random noise vector (Z)
# NOTE: This MUST match the number of subfolders you created in the 'train' directory.
NUM_CLASSES = 4   # We have 4 defect classes: captured, hole, horizontal, verticle
EPOCHS = 550     # Training epochs (GANs need many!)
BATCH_SIZE = 64
SAMPLE_INTERVAL = 50 # How often to save generated images
DATA_DIR = './defect_data/train' # Define the path where your training images are located

# --- 2. GENERATOR MODEL DEFINITION ---

def define_generator(latent_dim, n_classes):
    """
    Defines the Generator model (G) which takes noise (Z) and a class label (C)
    and outputs a synthetic image.
    """
    # 1. Condition Input (Class Label C)
    # The label is converted into a high-dimensional embedding.
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    # Scale the embedding up to match the start size for the latent noise
    n_nodes = 4 * 4 * 1  # 4x4 base size
    li = Dense(n_nodes)(li)
    # Reshape to a 4x4 feature map to concatenate with image features
    li = Reshape((4, 4, 1))(li)

    # 2. Latent Input (Noise Z)
    in_latent = Input(shape=(latent_dim,))
    # Scale the noise up to the starting dimensions for ConvTranspose
    n_nodes = 128 * 4 * 4 # Starting size for upsampling: 4x4
    gen = Dense(n_nodes)(in_latent)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((4, 4, 128))(gen) # 4x4x128 feature map

    # 3. Concatenate (C and Z)
    merge = Concatenate()([gen, li])

    # 4. Upsampling / Transposed Convolutional Layers
    # Start: 4x4x129 (128 + 1 from label)
    
    # Block 1: 4x4 -> 8x8
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    
    # Block 2: 8x8 -> 16x16
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # Block 3: 16x16 -> 32x32
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # Block 4: 32x32 -> 64x64
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
    # Block 5: 64x64 -> 128x128 (Output Layer)
    out_layer = Conv2DTranspose(CHANNELS, (4,4), strides=(2,2), padding='same', activation='tanh')(gen)

    # Define the model with two inputs
    model = Model([in_latent, in_label], out_layer, name='Generator')
    return model

# --- 3. DISCRIMINATOR MODEL DEFINITION ---

def define_discriminator(input_shape, n_classes):
    """
    Defines the Discriminator model (D) which takes an image (X) and a class label (C)
    and outputs a prediction of Real (1) or Fake (0).
    """
    # 1. Condition Input (Class Label C)
    in_label = Input(shape=(1,))
    # Convert label to a dense map matching the input image size (128x128)
    li = Embedding(n_classes, 128 * 128)(in_label)
    li = Reshape((128, 128, 1))(li) # 128x128x1 feature map

    # 2. Image Input (X)
    in_image = Input(shape=input_shape)
    
    # 3. Concatenate (X and C)
    merge = Concatenate()([in_image, li])

    # 4. Downsampling / Convolutional Layers
    # Start: 128x128x4 (3 channels for RGB + 1 channel for label map)

    # Block 1: 128x128 -> 64x64
    dis = Conv2D(64, (3,3), strides=(2,2), padding='same')(merge)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.4)(dis)
    
    # Block 2: 64x64 -> 32x32
    dis = Conv2D(64, (3,3), strides=(2,2), padding='same')(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.4)(dis)

    # Block 3: 32x32 -> 16x16
    dis = Conv2D(128, (3,3), strides=(2,2), padding='same')(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.4)(dis)
    
    # 5. Output Layer
    dis = Flatten()(dis)
    out_layer = Dense(1, activation='sigmoid')(dis) # 1 for Real, 0 for Fake

    # Define the model with two inputs
    model = Model([in_image, in_label], out_layer, name='Discriminator')
    # Compile the model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# --- 4. COMBINED CGAN MODEL DEFINITION ---

def define_cgan(g_model, d_model):
    """
    Defines the combined CGAN model. The Generator is trained via the Discriminator's output.
    The Discriminator's weights are frozen during CGAN training.
    """
    # Freeze Discriminator weights (only Generator is trained)
    d_model.trainable = False
    
    # Define CGAN inputs (same as Generator inputs)
    gen_noise_input = Input(shape=(LATENT_DIM,))
    gen_label_input = Input(shape=(1,))
    
    # Generate image
    gen_output = g_model([gen_noise_input, gen_label_input])
    
    # Classify output
    cgan_output = d_model([gen_output, gen_label_input])
    
    # Define the model
    model = Model([gen_noise_input, gen_label_input], cgan_output, name='CGAN')
    
    # Compile the model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# --- 5. DATA UTILITY FUNCTION (Real Data Loading) ---

def load_real_samples(directory=DATA_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Loads real training images and their labels using Keras utilities.
    Scales images to the required [-1, 1] range for GANs.
    """
    if not os.path.exists(directory):
        print(f"ERROR: Data directory not found at '{directory}'.")
        print("Please create the structure: './defect_data/train/[class_name]/'")
        return None, None

    print(f"Loading images from: {directory}")
    
    # We use a custom function for rescaling: (x / 127.5) - 1.0 -> scales to [-1, 1]
    def scale_to_tanh(image, label):
        image = tf.cast(image, tf.float32)
        # Normalize to [-1, 1]
        image = (image / 127.5) - 1.0 
        return image, label

    # Load data using Keras utility
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode='int', # Integer labels for class indices
        color_mode='rgb',
        image_size=target_size,
        interpolation='nearest',
        batch_size=None, # Load all data first
        shuffle=True
    )
    
    # Apply the scaling function
    dataset = dataset.map(scale_to_tanh)
    
    # Convert the dataset into numpy arrays for direct use in the training loop
    X_list, y_list = [], []
    for images, labels in dataset.as_numpy_iterator():
        X_list.append(images)
        y_list.append(labels)
        
    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1) # Reshape labels to (N, 1)

    print(f"Data Loaded: {X.shape[0]} samples, Image Shape: {X.shape[1:]}, Class Labels: {np.unique(y)}")
    return X, y.flatten()


def generate_latent_points(n_samples, n_classes, latent_dim):
    """Generates random noise vectors and associated random class labels."""
    z_input = np.random.randn(n_samples, latent_dim)
    # Generate random class labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]

def generate_fake_samples(g_model, n_samples, n_classes, latent_dim):
    """Uses the generator to create fake images and labels them as 0 (Fake)."""
    # Generate latent points and labels
    z_input, labels = generate_latent_points(n_samples, n_classes, latent_dim)
    # Generate images
    images = g_model.predict([z_input, labels], verbose=0)
    # Create labels for the Discriminator (0 for fake)
    y = np.zeros((n_samples, 1))
    return [images, labels], y

def generate_and_save_images(g_model, n_classes, latent_dim, epoch, examples=10):
    """Generates a sample batch of images from the Generator and saves them."""
    # Create output directory
    os.makedirs('generated_defects', exist_ok=True)
    
    # Generate fixed noise and class labels for consistent testing
    n_samples = n_classes 
    
    # Generate latent points and *fixed* labels for each class (0 to N-1)
    z_input = np.random.randn(n_samples, latent_dim)
    labels = np.arange(0, n_classes) 
    
    # Generate images
    X = g_model.predict([z_input, labels], verbose=0)
    # Rescale images back to 0-1 range for plotting
    X = (X + 1) / 2.0
    
    plt.figure(figsize=(n_classes * 2, 2))
    plt.suptitle(f"Epoch {epoch} - Synthetic Defects by Class")
    
    for i in range(n_samples):
        plt.subplot(1, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X[i])
        plt.title(f"Class {labels[i]}")
        
    filename = f'generated_defects/generated_plot_e{epoch:04d}.png'
    plt.savefig(filename)
    plt.close()
    print(f'>Saved output image: {filename}')


# --- 6. TRAINING FUNCTION ---

def train_cgan(g_model, d_model, cgan_model, dataset, n_epochs=EPOCHS, n_batch=BATCH_SIZE):
    """The main training loop for the CGAN."""
    # Unpack the dataset
    X_train, y_train = dataset
    
    # Handle empty dataset if loading failed
    if X_train is None or y_train is None:
        print("\nFATAL ERROR: Training dataset is empty. Cannot start training.")
        return
        
    # Calculate the number of batches per epoch
    bat_per_epo = int(X_train.shape[0] / n_batch)
    # Half batch size for training D on real and fake samples
    half_batch = n_batch // 2
    
    print(f"Starting CGAN Training: {n_epochs} epochs, {bat_per_epo} batches/epoch.")

    for i in range(n_epochs):
        for j in range(bat_per_epo):
            # --- PHASE 1: Train the Discriminator (D) ---
            
            # Get real samples
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            X_real, labels_real = X_train[idx], y_train[idx]
            y_real = np.ones((half_batch, 1)) # Labels are 1 (Real)
            
            # Generate fake samples
            [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, half_batch, NUM_CLASSES, LATENT_DIM) # Labels are 0 (Fake)
            
            # Combine real and fake samples
            X_D, y_D, labels_D = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake)), np.hstack((labels_real, labels_fake))
            
            # Train Discriminator
            d_loss, d_acc = d_model.train_on_batch([X_D, labels_D], y_D)

            # --- PHASE 2: Train the Generator (G) via the CGAN ---
            
            # Prepare points in latent space for Generator
            [z_input, labels_gen] = generate_latent_points(n_batch, NUM_CLASSES, LATENT_DIM)
            # Create inverted labels (we want D to classify these as REAL)
            y_gan = np.ones((n_batch, 1))
            
            # Train Generator
            g_loss = cgan_model.train_on_batch([z_input, labels_gen], y_gan)
            
            # Print progress
            print(f'>Epoch {i+1}, Batch {j+1}/{bat_per_epo}, D_Loss={d_loss:.4f}, D_Acc={d_acc:.2f}, G_Loss={g_loss:.4f}')

        # Summarize performance and save generated images
        if (i+1) % SAMPLE_INTERVAL == 0:
            generate_and_save_images(g_model, NUM_CLASSES, LATENT_DIM, i+1)
            
# --- 7. MAIN EXECUTION ---

if __name__ == '__main__':
    # 1. Define Models
    d_model = define_discriminator((IMG_HEIGHT, IMG_WIDTH, CHANNELS), NUM_CLASSES)
    g_model = define_generator(LATENT_DIM, NUM_CLASSES)
    cgan_model = define_cgan(g_model, d_model)

    print("\n--- Conditional GAN Architecture Summary ---")
    g_model.summary()
    d_model.summary()
    cgan_model.summary()
    print("------------------------------------------\n")
    
    # 2. Load Real Dataset
    # This function uses your defect_data/train/[class] folder structure.
    dataset = load_real_samples() 

    # 3. Train the CGAN
    train_cgan(g_model, d_model, cgan_model, dataset)

    print("\nCGAN training pipeline execution complete.")