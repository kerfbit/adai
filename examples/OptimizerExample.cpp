#include <iomanip>
#include <iostream>
#include "Optimizer.hpp"

/**
 * @file OptimizerExample.cpp
 * @brief Demonstrates usage of the centralized Optimizer class
 *
 * This example shows:
 * 1. Creating optimizers with different algorithms
 * 2. Adding parameter groups
 * 3. Gradient clipping
 * 4. Optimization step
 * 5. Monitoring gradient statistics
 */

void print_matrix(const Matrix& m, const std::string& name) {
    std::cout << name << " (" << m.rows << "x" << m.cols << "):\n";
    for (int i = 0; i < std::min(3, m.rows); i++) {
        std::cout << "  ";
        for (int j = 0; j < std::min(5, m.cols); j++) {
            std::cout << std::fixed << std::setprecision(4) << m(i, j) << " ";
        }
        if (m.cols > 5)
            std::cout << "...";
        std::cout << "\n";
    }
    if (m.rows > 3)
        std::cout << "  ...\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   Optimizer Example - Gradient Updates  ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    // Create sample weight matrices
    std::cout << "1️⃣  Creating sample parameters...\n";
    Matrix weights1(3, 4);
    Matrix gradients1(3, 4);

    Matrix weights2(2, 3);
    Matrix gradients2(2, 3);

    // Initialize with some values
    for (int i = 0; i < weights1.rows; i++) {
        for (int j = 0; j < weights1.cols; j++) {
            weights1(i, j) = 0.5f + 0.1f * (i + j);
            gradients1(i, j) = 0.01f * (i - j);
        }
    }

    for (int i = 0; i < weights2.rows; i++) {
        for (int j = 0; j < weights2.cols; j++) {
            weights2(i, j) = 0.3f + 0.05f * (i + j);
            gradients2(i, j) = 0.02f * (j - i);
        }
    }

    print_matrix(weights1, "Initial weights1");
    print_matrix(gradients1, "Gradients1");
    std::cout << "\n";

    // ========================================
    // Example 1: SGD Optimizer
    // ========================================
    std::cout << "2️⃣  Testing SGD Optimizer\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    Optimizer sgd_optimizer(OptimizerType::SGD, 0.1f);
    sgd_optimizer.add_parameter_group(&weights1, &gradients1);
    sgd_optimizer.add_parameter_group(&weights2, &gradients2);

    std::cout << "Optimizer: " << sgd_optimizer.get_optimizer_name() << "\n";
    std::cout << "Learning rate: " << sgd_optimizer.get_learning_rate() << "\n";
    std::cout << "Total parameters: " << sgd_optimizer.total_parameters() << "\n";
    std::cout << "Parameter groups: " << sgd_optimizer.num_parameters() << "\n\n";

    float grad_norm = sgd_optimizer.get_gradient_norm();
    std::cout << "Gradient norm before step: " << grad_norm << "\n";

    sgd_optimizer.step();

    print_matrix(weights1, "Weights1 after SGD step");
    std::cout << "\n";

    // ========================================
    // Example 2: Adam Optimizer with Gradient Clipping
    // ========================================
    std::cout << "3️⃣  Testing Adam Optimizer with Gradient Clipping\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    // Reset weights
    for (int i = 0; i < weights1.rows; i++) {
        for (int j = 0; j < weights1.cols; j++) {
            weights1(i, j) = 0.5f + 0.1f * (i + j);
            gradients1(i, j) = 2.0f * (i - j);  // Larger gradients
        }
    }

    Optimizer adam_optimizer(OptimizerType::ADAM, 0.001f);
    adam_optimizer.add_parameter_group(&weights1, &gradients1);
    adam_optimizer.set_max_grad_norm(1.0f);  // Clip to max norm of 1.0

    std::cout << "Optimizer: " << adam_optimizer.get_optimizer_name() << "\n";
    std::cout << "Learning rate: " << adam_optimizer.get_learning_rate() << "\n";
    std::cout << "Max gradient norm: 1.0\n\n";

    grad_norm = adam_optimizer.get_gradient_norm();
    std::cout << "Gradient norm before clipping: " << grad_norm << "\n";

    float clipped_norm = adam_optimizer.clip_gradients();
    std::cout << "Gradient norm after clipping: " << adam_optimizer.get_gradient_norm() << "\n";
    std::cout << "Clipping applied: " << (clipped_norm > 1.0f ? "Yes" : "No") << "\n\n";

    adam_optimizer.step();

    print_matrix(weights1, "Weights1 after Adam step (with clipping)");
    std::cout << "\n";

    // ========================================
    // Example 3: AdamW with Weight Decay
    // ========================================
    std::cout << "4️⃣  Testing AdamW with Weight Decay\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    // Reset weights
    for (int i = 0; i < weights1.rows; i++) {
        for (int j = 0; j < weights1.cols; j++) {
            weights1(i, j) = 0.5f + 0.1f * (i + j);
            gradients1(i, j) = 0.01f * (i - j);
        }
    }

    Optimizer adamw_optimizer(OptimizerType::ADAMW, 0.001f);
    adamw_optimizer.add_parameter_group(&weights1, &gradients1);
    adamw_optimizer.set_weight_decay(0.01f);  // L2 regularization
    adamw_optimizer.set_betas(0.9f, 0.999f);

    std::cout << "Optimizer: " << adamw_optimizer.get_optimizer_name() << "\n";
    std::cout << "Learning rate: " << adamw_optimizer.get_learning_rate() << "\n";
    std::cout << "Weight decay: 0.01\n";
    std::cout << "Betas: (0.9, 0.999)\n\n";

    adamw_optimizer.step();

    print_matrix(weights1, "Weights1 after AdamW step");
    std::cout << "\n";

    // ========================================
    // Example 4: Multiple Steps with Learning Rate Scheduling
    // ========================================
    std::cout << "5️⃣  Training Loop Simulation (5 steps)\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    Optimizer train_optimizer(OptimizerType::ADAM, 0.01f);
    train_optimizer.add_parameter_group(&weights1, &gradients1);
    train_optimizer.set_max_grad_norm(1.0f);

    for (int step = 0; step < 5; step++) {
        // Simulate gradient computation
        for (int i = 0; i < gradients1.rows; i++) {
            for (int j = 0; j < gradients1.cols; j++) {
                gradients1(i, j) = 0.1f * (std::rand() % 100 - 50) / 50.0f;
            }
        }

        float g_norm = train_optimizer.get_gradient_norm();
        train_optimizer.clip_gradients();
        train_optimizer.step();

        std::cout << "Step " << (step + 1) << ": ";
        std::cout << "grad_norm=" << std::fixed << std::setprecision(4) << g_norm;
        std::cout << ", lr=" << train_optimizer.get_learning_rate() << "\n";

        // Simulate learning rate decay
        if ((step + 1) % 2 == 0) {
            float new_lr = train_optimizer.get_learning_rate() * 0.9f;
            train_optimizer.set_learning_rate(new_lr);
        }

        train_optimizer.zero_grad();
    }

    std::cout << "\n";
    print_matrix(weights1, "Final weights after 5 steps");

    // ========================================
    // Summary
    // ========================================
    std::cout << "\n╔══════════════════════════════════════════╗\n";
    std::cout << "║            Example Complete              ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";
    std::cout << "\nKey Features Demonstrated:\n";
    std::cout << "  ✓ Multiple optimizer types (SGD, Adam, AdamW)\n";
    std::cout << "  ✓ Gradient clipping for training stability\n";
    std::cout << "  ✓ Weight decay / L2 regularization\n";
    std::cout << "  ✓ Learning rate scheduling\n";
    std::cout << "  ✓ Gradient norm monitoring\n";
    std::cout << "  ✓ Multiple parameter groups\n";

    return 0;
}
