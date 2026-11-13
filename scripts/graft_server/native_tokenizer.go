// Built into the same pinned Ollama executable as the serving runner.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/template"
)

func graftTokenize() error {
	if len(os.Args) != 4 {
		return fmt.Errorf("usage: ollama graft-tokenize MODEL TEMPLATE")
	}
	raw, err := os.ReadFile(os.Args[3])
	if err != nil {
		return err
	}
	tmpl, err := template.Parse(string(raw))
	if err != nil {
		return err
	}
	llama.BackendInit()
	model, err := llama.LoadModelFromFile(os.Args[2], llama.ModelParams{VocabOnly: true})
	if err != nil {
		return err
	}
	defer llama.FreeModel(model)
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 65536), 4*1024*1024)
	out := json.NewEncoder(os.Stdout)
	for in.Scan() {
		var messages []api.Message
		if err := json.Unmarshal(in.Bytes(), &messages); err != nil {
			return err
		}
		if len(messages) != 2 || messages[0].Role != "system" || messages[1].Role != "user" ||
			messages[0].Content == "" || messages[1].Content == "" ||
			len(messages[0].Images)+len(messages[1].Images)+len(messages[0].ToolCalls)+len(messages[1].ToolCalls) != 0 {
			return fmt.Errorf("expected two no-tools text messages")
		}
		var prompt bytes.Buffer
		if err := tmpl.Execute(&prompt, template.Values{Messages: messages, IsThinkSet: true}); err != nil {
			return err
		}
		// The compatibility runner's inputs() uses the same BOS and special-token flags.
		ids, err := model.Tokenize(prompt.String(), true, true)
		if err != nil {
			return err
		}
		if err := out.Encode(struct {
			Rendered string `json:"rendered"`
			IDs      []int  `json:"token_ids"`
		}{prompt.String(), ids}); err != nil {
			return err
		}
	}
	return in.Err()
}
