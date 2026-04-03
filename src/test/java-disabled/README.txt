Эти тесты временно вне src/test/java: они описывают API full-GPU / device logits / device decoder backward,
который ещё не в main. Maven их не компилирует — зато основной mvn test снова зелёный.

Вернуть: перенести .java обратно в src/test/java/.../ те же пакеты и дописать production-код.
