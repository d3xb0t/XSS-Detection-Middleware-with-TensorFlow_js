/**
 * XSS Detection Middleware using TensorFlow.js
 * @module xssDetector
 * @description Implements a deep learning model to detect XSS attacks in HTTP requests
 *              using character-level CNN (Convolutional Neural Network)
 * @author d3xb0t
 * @version 1.0.0
 * @license MIT
 * @created 2023-11-15
 * @lastModified 2023-11-15
 */

// Import TensorFlow.js for Node.js environment
import * as tf from '@tensorflow/tfjs-node';

// =============================================
// DATA DEFINITION SECTION
// =============================================

/**
 * @constant {Array<string>} malicious_samples
 * @description Comprehensive collection of known XSS attack patterns and SQL injection samples
 * @note Includes:
 *        - Basic script tag injections
 *        - Event handler based XSS (onerror, onload, etc.)
 *        - SVG-based XSS vectors
 *        - CSS expression attacks
 *        - HTML5-based vectors (details, audio, video)
 *        - Obfuscated attacks (hex encoding, nested tags)
 *        - SQL injection samples
 */
const malicious_samples = [
    "<script>alert('XSS')<\/script>",
    `"><script>alert('XSS')<\/script>`,
    `"><img src=x onerror=alert('XSS')>`,
    `"><svg/onload=alert(1)>`,
    `--!>\x3Cstyle/onload=alert(1)//`,
    `'><audio src/onerror=alert(1)>`,
    `"><video><source onerror="javascript:alert(1)">`,
    `<body onpageshow="alert(1)">`,
    `<img src="x:?onerror=alert(1)">`,
    `<div style="width: expression(alert('XSS'));">`,
    `<\/script><script>alert(1)<\/script>`,
    `"><iframe src="javascript:alert('XSS')"><\/iframe>`,
    `' OR 1=1 --`,
    `1; DROP TABLE users--`,
    `<img src=x onmouseover=alert(1)>`,
    `<a href="javascript:alert('XSS')">Click me<\/a>`,
    `--><script>alert(1)<\/script>`,
    `\";alert(1)//`,
    `<\/title><script>alert(1)<\/script>`,
    `<style>@import "javascript:alert(1)";<\/style>`,
    `<img src="javascript:alert('XSS');">`,
    `"><body onload=alert('XSS')>`,
    `<details/open/ontoggle=alert(1)>`,
    `<object data="javascript:alert(1)"><\/object>`,
    `<iframe srcdoc="<script>alert(1)<\/script>"><\/iframe>`,
    `' onfocus=alert(1) autofocus`,
    `'--><\/style><\/script><script>alert(1)<\/script>`,
    `<scr<script>ipt>alert(1)<\/scr<\/script>ipt>`,
    `%3cscript%3ealert(1)%3c/script%3e`
];

/**
 * @constant {Array<string>} benign_samples
 * @description Collection of normal, safe input samples for training
 * @note Includes:
 *        - Regular text sentences
 *        - Common user inputs (emails, phone numbers)
 *        - Safe special characters
 *        - JSON-like strings
 *        - Numeric inputs
 *        - Common web application inputs
 */
const benign_samples = [
    "Hello world",
    "This is a normal search query.",
    "user@example.com",
    "123-456-7890",
    "A sentence with & and = symbols.",
    "Just a simple string",
    "Another/safe/path",
    "{\"key\": \"value\"}",
    "A string with some numbers 12345",
    "Final benign test.",
    "The weather is nice today.",
    "I need to buy groceries.",
    "What time is the meeting?",
    "Please review this document.",
    "The report is due on Friday.",
    "This is a normal sentence.",
    "My name is John Doe.",
    "What is your favorite color?",
    "The quick brown fox jumps over the lazy dog.",
    "User-12345",
    "password123",
    "123 Main Street, Anytown",
    "This is a test.",
    "Another safe string.",
    "Just some regular text here.",
    "No scripts, no problem.",
    "How are you today?",
    "This is a product description.",
    "Search for items.",
    "Login successful.",
    "Update profile information.",
    "My cart has 5 items.",
    "Contact us for more information.",
    "About our company.",
    "Frequently Asked Questions",
    "Terms of Service",
    "Privacy Policy",
    "A simple text without any special characters.",
    "Another example of a safe input."
];

// =============================================
// DATA PREPROCESSING SECTION
// =============================================

/**
 * @constant {Array<string>} all_chars
 * @description Unique characters from both malicious and benign samples
 * @note Used to build character-level vocabulary for the model
 *       Includes all ASCII and Unicode characters found in samples
 */
const all_chars = Array.from(new Set([...malicious_samples.join(''), ...benign_samples.join('')]));

/**
 * @constant {Object} char_to_index
 * @description Mapping of characters to numerical indices for model input
 * @property {number} [key: string] - Character to index mapping
 * @note Index starts from 1 (0 is reserved for padding/unknown characters)
 *       This follows standard NLP vocabulary indexing practices
 */
const char_to_index = {};
all_chars.forEach((char, i) => char_to_index[char] = i + 1);

// =============================================
// MODEL CONFIGURATION SECTION
// =============================================

/**
 * @constant {number} VOCAB_SIZE
 * @description Size of the character vocabulary (+1 for padding/unknown)
 * @note Determines the input dimension for the embedding layer
 */
const VOCAB_SIZE = Object.keys(char_to_index).length + 1;

/**
 * @constant {number} MAX_LEN
 * @description Maximum length of input sequences
 * @note Longer sequences will be truncated, shorter ones will be padded
 *       Chosen based on analysis of typical XSS attack lengths
 */
const MAX_LEN = 100;

// =============================================
// UTILITY FUNCTIONS SECTION
// =============================================

/**
 * Pads an array to specified length with given value
 * @function pad
 * @param {Array} arr - Input array to be padded
 * @param {number} len - Target length of the array
 * @param {*} [val=0] - Padding value (default: 0)
 * @returns {Array} New array with length = len
 * @note Standard preprocessing for neural network inputs
 *       Ensures uniform input dimensions
 */
const pad = (arr, len, val = 0) => arr.concat(Array(len - arr.length).fill(val));

/**
 * Converts text to numerical sequence using character indices
 * @function textToSequence
 * @param {string} text - Input text to convert
 * @returns {tf.Tensor} 2D tensor of shape [1, MAX_LEN] containing the sequence
 * @note Performs:
 *       1. Character to index mapping
 *       2. Sequence truncation to MAX_LEN
 *       3. Padding to MAX_LEN
 */
function textToSequence(text) {
    let sequence = text.split('').map(char => char_to_index[char] || 0);
    const paddedSequence = pad(sequence.slice(0, MAX_LEN), MAX_LEN);
    return tf.tensor2d([paddedSequence], [1, MAX_LEN]);
}

// =============================================
// MODEL DEFINITION SECTION
// =============================================

/**
 * @variable {tf.Sequential} model
 * @description The XSS detection model instance
 * @note Model architecture:
 *       - Embedding Layer: Character to vector mapping
 *       - Conv1D Layer: Local pattern detection
 *       - GlobalMaxPooling: Dimensionality reduction
 *       - Dense Layers: Classification
 */
let model;

/**
 * Trains the XSS detection model
 * @async
 * @function trainModel
 * @description Creates and trains a CNN model for XSS detection
 * @returns {Promise<void>}
 * @throws {Error} If training fails
 * @note Training process:
 *       1. Prepares training data (malicious + benign samples)
 *       2. Creates model architecture
 *       3. Compiles model with appropriate loss and metrics
 *       4. Trains model for 100 epochs
 */
async function trainModel() {
    console.log('Starting XSS detection model training...');

    try {
        // Prepare training data
        const malicious_tensors = malicious_samples.map(text => textToSequence(text).arraySync()[0]);
        const benign_tensors = benign_samples.map(text => textToSequence(text).arraySync()[0]);

        const x_train_data = malicious_tensors.concat(benign_tensors);
        const y_train_data = Array(malicious_samples.length).fill(1).concat(Array(benign_samples.length).fill(0));

        // Convert to tensors
        const x_train = tf.tensor2d(x_train_data, [x_train_data.length, MAX_LEN]);
        const y_train = tf.tensor2d(y_train_data, [y_train_data.length, 1]);

        // =============================================
        // MODEL ARCHITECTURE DEFINITION
        // =============================================
        
        model = tf.sequential();
        
        /**
         * Embedding Layer
         * - Converts character indices to dense vectors
         * - Learns character-level representations
         * - Input: [batch_size, MAX_LEN]
         * - Output: [batch_size, MAX_LEN, 32]
         */
        model.add(tf.layers.embedding({
            inputDim: VOCAB_SIZE,
            outputDim: 32,
            inputLength: MAX_LEN,
            name: 'embedding_layer'
        }));
        
        /**
         * Convolutional Layer
         * - Extracts local character patterns
         * - Uses 64 filters with window size 5
         * - ReLU activation for non-linearity
         * - Output: [batch_size, MAX_LEN - kernelSize + 1, 64]
         */
        model.add(tf.layers.conv1d({
            filters: 64,
            kernelSize: 5,
            activation: 'relu',
            name: 'conv1d_layer'
        }));
        
        /**
         * Global Max Pooling Layer
         * - Reduces sequence dimension by taking maximum values
         * - Output: [batch_size, 64]
         */
        model.add(tf.layers.globalMaxPooling1d({
            name: 'global_max_pooling'
        }));
        
        /**
         * Dense Layer
         * - 32 units with ReLU activation
         * - Learns higher-level features
         * - Output: [batch_size, 32]
         */
        model.add(tf.layers.dense({ 
            units: 32, 
            activation: 'relu',
            name: 'dense_layer_1'
        }));
        
        /**
         * Output Layer
         * - Single unit with sigmoid activation
         * - Produces probability score (0-1)
         * - Output: [batch_size, 1]
         */
        model.add(tf.layers.dense({ 
            units: 1, 
            activation: 'sigmoid',
            name: 'output_layer'
        }));

        // =============================================
        // MODEL COMPILATION
        // =============================================
        
        /**
         * Model Compilation
         * - Adam optimizer: Adaptive learning rate
         * - Binary crossentropy: Suitable for binary classification
         * - Accuracy metric: For monitoring training progress
         */
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy'],
        });

        // =============================================
        // MODEL TRAINING
        // =============================================
        
        console.log('Starting model training...');
        await model.fit(x_train, y_train, {
            epochs: 100,
            batchSize: 16,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if ((epoch + 1) % 10 === 0) {
                        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}`);
                    }
                }
            }
        });

        console.log('Model training completed successfully.');
    } catch (error) {
        console.error('Error during model training:', error);
        throw error;
    } finally {
        // Clean up tensors to avoid memory leaks
        tf.dispose([x_train, y_train]);
    }
}

// Start training (fire and forget)
trainModel().catch(console.error);

// =============================================
// MIDDLEWARE IMPLEMENTATION SECTION
// =============================================

/**
 * XSS Detection Middleware
 * @function xssMiddleware
 * @description Middleware function to detect XSS attacks in incoming requests
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next middleware function
 * @returns {void|Object} Either passes to next middleware or returns 403 response
 * @note Operation flow:
 *       1. Checks if model is loaded
 *       2. Extracts all possible inputs from request
 *       3. For each input:
 *          a. Converts to sequence
 *          b. Gets model prediction
 *          c. Blocks request if XSS detected (score > 0.7)
 *       4. Passes to next middleware if all inputs are safe
 */
const xssMiddleware = (req, res, next) => {
    // Check if model is ready
    if (!model) {
        console.warn("XSS Detection: Model not trained yet, skipping validation");
        return next();
    }

    try {
        // Extract all possible inputs from request
        const inputs = [
            ...Object.values(req.query || {}),
            ...Object.values(req.body || {}),
            ...Object.values(req.params || {})
        ];

        // Process each input
        for (const input of inputs) {
            if (typeof input === 'string' && input.trim().length > 0) {
                const sequence = textToSequence(input);
                const prediction = model.predict(sequence);
                const score = prediction.dataSync()[0];

                console.log(`XSS Detection: Input "${input.substring(0, 30)}..." Score: ${score.toFixed(4)}`);

                // Dispose tensors to avoid memory leaks
                tf.dispose([sequence, prediction]);

                // Block request if XSS detected
                if (score > 0.7) {
                    console.warn(`XSS Attack Detected: ${input.substring(0, 50)}...`);
                    return res.status(403).json({
                        error: 'Forbidden',
                        message: 'Potential XSS attack detected',
                        statusCode: 403
                    });
                }
            }
        }

        // All inputs are safe
        next();
    } catch (error) {
        console.error('XSS Detection Error:', error);
        // Fail safely - allow request to proceed
        next();
    }
};

// Export middleware
export default xssMiddleware;
