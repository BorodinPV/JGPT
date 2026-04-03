package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.core.Tensor;

public class AdamOptimizerTest {
    public static void main(String[] args) {
        System.out.println("🧪 Testing Adam Optimizer...");

        // Простая задача: минимизировать f(x) = x²
        // Минимум при x = 0

        Tensor x = new Tensor(new int[]{1});
        x.internalBuffer()[0] = 5.0f;  // Начальное значение

        // ВАЖНО: используем больший learning rate для теста!
        AdamOptimizer optimizer = AdamOptimizer.forTesting();

        System.out.printf("Initial x: %.4f%n", x.internalBuffer()[0]);
        System.out.printf("Learning rate: 0.1 (for testing)%n");
        System.out.println();

        // 500 шагов градиентного спуска
        for (int i = 0; i < 500; i++) {
            // Градиент f(x) = x² → df/dx = 2x
            x.zeroGrad();
            Tensor grad = new Tensor(new int[]{1});
            grad.zeroGrad();
            grad.gradBuffer()[0] = 2 * x.internalBuffer()[0];

            optimizer.step(x, grad);

            if (i % 100 == 0) {
                float val = x.internalBuffer()[0];
                System.out.printf("Step %d: x = %.6f, loss = %.6f%n",
                        i, val, val * val);
            }
        }

        System.out.println();
        System.out.printf("Final x: %.6f (expected ~0.0)%n", x.internalBuffer()[0]);
        System.out.printf("Final loss: %.6f (expected ~0.0)%n",
                x.internalBuffer()[0] * x.internalBuffer()[0]);

        if (Math.abs(x.internalBuffer()[0]) < 0.1f) {
            System.out.println("✅ Adam optimizer test: PASS");
        } else {
            System.out.println("⚠️  Adam optimizer test: x didn't converge to 0");
            System.out.println("   Tip: Try higher learning rate or more steps");
        }

        // Тест градиентного клиппинга
        System.out.println("\n🧪 Testing Gradient Clipping...");
        Tensor largeGrad = new Tensor(new int[]{100});
        largeGrad.zeroGrad();
        for (int i = 0; i < 100; i++) {
            largeGrad.gradBuffer()[i] = 10.0f;  // Большие градиенты
        }

        float norm = optimizer.clipGradients(
                new java.util.ArrayList<Tensor>() {{ add(largeGrad); }},
                1.0f
        );
        System.out.printf("Gradient norm before clip: %.2f%n", norm);

        // После клиппинга норма должна быть <= 1.0
        float normAfter = 0;
        for (float g : largeGrad.gradBuffer()) {
            normAfter += g * g;
        }
        normAfter = (float) Math.sqrt(normAfter);
        System.out.printf("Gradient norm after clip: %.2f (should be ≤ 1.0)%n", normAfter);

        if (normAfter <= 1.01f) {
            System.out.println("✅ Gradient clipping test: PASS");
        } else {
            System.out.println("❌ Gradient clipping test: FAIL");
        }

        System.out.println("\n✅ Adam Optimizer test complete!");
    }
}