package com.veles.llm.jgpt.data;

/** Одна реплика диалога для SFT. */
public final class SftTurn {

    public enum Role {
        USER,
        ASSISTANT
    }

    public final Role role;
    public final String content;

    public SftTurn(Role role, String content) {
        this.role = role;
        this.content = content;
    }
}
