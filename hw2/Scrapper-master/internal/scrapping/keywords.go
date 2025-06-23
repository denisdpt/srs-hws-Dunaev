package scrapping

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

var keyWordsServiceURL string

func init() {
	keyWordsServiceURL = os.Getenv("KEYWORDS_SERVICE_URL")
	if keyWordsServiceURL == "" {
		keyWordsServiceURL = "http://localhost:5000/extract"
	}
}

type ExtractRequest struct {
	Text string `json:"text"`
}

type ExtractResponse struct {
	Keywords []string `json:"keywords"`
	Error    string   `json:"error"`
}

func (s *Service) getKeywords(text string) ([]string, error) {
	requestData := ExtractRequest{
		Text: text,
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return nil, fmt.Errorf("json.Marshal: %w", err)
	}

	resp, err := http.Post(keyWordsServiceURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("http.Post: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("io.ReadAll: %w", err)
	}

	var response ExtractResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("json.Unmarshal: %w", err)
	}

	if response.Error != "" {
		return nil, fmt.Errorf("keywordsService: %s", response.Error)
	}

	return response.Keywords, nil
}
