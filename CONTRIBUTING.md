# Contributing to JGPT / Участие в разработке JGPT

Thank you for your interest in contributing to JGPT!  
Спасибо за интерес к участию в разработке JGPT!

---

## Development Setup / Настройка окружения

### Requirements / Требования
- Java 25+ with Vector API / Java 25+ с Vector API
- CUDA 12.x or 13.x with cuBLAS / CUDA 12.x или 13.x с cuBLAS
- Linux: GCC ≤ 13 (or `-allow-unsupported-compiler` for GCC 14+) / GCC ≤ 13
- Windows: Visual Studio 2022 Build Tools (MSVC) + CMake (`winget install Kitware.CMake`)
- Maven 3.9+

### Build / Сборка
```bash
# Clone repository / Клонирование репозитория
git clone <repository-url>
cd JGPT

# Native CUDA/JNI — Linux
./scripts/linux/build-cuda.sh

# Native CUDA/JNI — Windows (PowerShell)
.\scripts\windows\build-cuda.ps1
. .\build\jgpt-cuda-env.ps1

# Build Java code / Сборка Java кода
mvn compile

# Run tests / Запуск тестов
mvn test
```

Launcher map: [scripts/README.md](scripts/README.md).

### Build with custom FlashAttention tile size / Сборка с кастомным размером плитки FlashAttention
```bash
# For GPUs with limited shared memory (RTX 3080)
# Для GPU с ограниченной shared memory (RTX 3080)
cd build
JGPT_FA_TILE_SIZE=128 cmake ..
cmake --build .
```

---

## Code Style / Стиль кода

### Java
- Follow standard Java conventions / Следуйте стандартным соглашениям Java
- Use `final` where possible / Используйте `final` где возможно
- Document public APIs with Javadoc / Документируйте публичные API с помощью Javadoc

### C++/CUDA
- Use `constexpr` for compile-time constants / Используйте `constexpr` для констант времени компиляции
- Prefer `__restrict__` for pointer parameters / Предпочитайте `__restrict__` для параметров-указателей
- Document kernels with block/thread dimensions / Документируйте ядра с размерностями блоков/потоков

---

## Testing / Тестирование

Before submitting PR / Перед отправкой PR:
1. Run full training for at least 100 steps / Запустите полное обучение минимум на 100 шагов
2. Verify no CUDA errors in logs / Проверьте отсутствие ошибок CUDA в логах
3. Check performance metrics (should be ~25k+ tokens/s on RTX 3080) / Проверьте метрики производительности (должно быть ~25k+ токенов/сек на RTX 3080)

---

## Submitting Changes / Отправка изменений

1. Fork the repository / Сделайте форк репозитория
2. Create a feature branch (`git checkout -b feature/amazing-feature`) / Создайте ветку фичи
3. Commit your changes (`git commit -m 'Add amazing feature'`) / Закоммитьте изменения
4. Push to the branch (`git push origin feature/amazing-feature`) / Запушьте ветку
5. Open a Pull Request / Откройте Pull Request

---

## Reporting Issues / Сообщение об ошибках

When reporting bugs, please include / При сообщении об ошибках, пожалуйста, укажите:
- GPU model and CUDA version / Модель GPU и версию CUDA
- Java version (`java -version`) / Версию Java
- GCC version (`gcc --version`) / Версию GCC
- Error messages from `training_allbooks.log` / Сообщения об ошибках из `training_allbooks.log`
- Steps to reproduce / Шаги для воспроизведения

---

## Questions? / Вопросы?

Join discussions in GitHub Issues or check the documentation in `docs/`.  
Присоединяйтесь к обсуждениям в GitHub Issues или смотрите документацию в `docs/`.
