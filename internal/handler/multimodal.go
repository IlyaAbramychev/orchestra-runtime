package handler

import (
	"fmt"
	"strings"

	"github.com/operium/orchestra-runtime/internal/model"
)

const (
	maxImagesPerRequest       = 16
	maxDecodedImageBytes      = 20 * 1024 * 1024
	maxDecodedImageBytesTotal = 50 * 1024 * 1024
)

var supportedImageMIMETypes = map[string]struct{}{
	"image/jpeg": {},
	"image/png":  {},
	"image/webp": {},
}

func validateMultimodalMessages(messages []model.ChatMessage) error {
	imageCount := 0
	totalBytes := 0
	for messageIndex, message := range messages {
		if len(message.Parts) > 0 && len(message.Images) > 0 {
			return &badRequestErr{fmt.Sprintf("messages[%d] cannot combine content image parts with images", messageIndex)}
		}
		for partIndex, part := range message.Parts {
			if part.Type != "image_url" {
				continue
			}
			if part.ImageDetail != "" && part.ImageDetail != "auto" && part.ImageDetail != "low" && part.ImageDetail != "high" {
				return &badRequestErr{fmt.Sprintf("messages[%d].content[%d].image_url.detail must be auto, low, or high", messageIndex, partIndex)}
			}
			size, err := validateImagePayload(part.ImageURL, true)
			if err != nil {
				return &badRequestErr{fmt.Sprintf("messages[%d].content[%d]: %v", messageIndex, partIndex, err)}
			}
			imageCount++
			totalBytes += size
		}
		for imageIndex, image := range message.Images {
			size, err := validateImagePayload(image, false)
			if err != nil {
				return &badRequestErr{fmt.Sprintf("messages[%d].images[%d]: %v", messageIndex, imageIndex, err)}
			}
			imageCount++
			totalBytes += size
		}
	}
	if imageCount > maxImagesPerRequest {
		return &badRequestErr{fmt.Sprintf("image count %d exceeds limit %d", imageCount, maxImagesPerRequest)}
	}
	if totalBytes > maxDecodedImageBytesTotal {
		return &badRequestErr{fmt.Sprintf("total decoded image size %d exceeds limit %d", totalBytes, maxDecodedImageBytesTotal)}
	}
	return nil
}

func validateImagePayload(raw string, requireDataURI bool) (int, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return 0, fmt.Errorf("empty image payload")
	}
	payload := value
	if strings.HasPrefix(strings.ToLower(value), "data:") {
		comma := strings.IndexByte(value, ',')
		if comma <= len("data:") {
			return 0, fmt.Errorf("invalid image data URI")
		}
		metadata := strings.ToLower(value[len("data:"):comma])
		fields := strings.Split(metadata, ";")
		mimeType := strings.TrimSpace(fields[0])
		if _, ok := supportedImageMIMETypes[mimeType]; !ok {
			return 0, fmt.Errorf("unsupported image format %q", mimeType)
		}
		base64Encoded := false
		for _, field := range fields[1:] {
			if strings.TrimSpace(field) == "base64" {
				base64Encoded = true
				break
			}
		}
		if !base64Encoded {
			return 0, fmt.Errorf("image data URI must use base64 encoding")
		}
		payload = value[comma+1:]
	} else if requireDataURI {
		if strings.HasPrefix(value, "http://") || strings.HasPrefix(value, "https://") {
			return 0, fmt.Errorf("remote image URLs are not supported")
		}
		return 0, fmt.Errorf("image_url must be a base64 data URI")
	}
	if payload == "" {
		return 0, fmt.Errorf("empty image payload")
	}
	if strings.IndexAny(payload, " \t\r\n") >= 0 {
		return 0, fmt.Errorf("invalid base64 image: whitespace is not allowed")
	}
	decodedSize := estimatedBase64DecodedSize(payload)
	if decodedSize > maxDecodedImageBytes {
		return 0, fmt.Errorf("decoded image size %d exceeds limit %d", decodedSize, maxDecodedImageBytes)
	}
	return decodedSize, nil
}

func estimatedBase64DecodedSize(payload string) int {
	size := len(payload) * 3 / 4
	if strings.HasSuffix(payload, "==") {
		size -= 2
	} else if strings.HasSuffix(payload, "=") {
		size--
	}
	if size < 0 {
		return 0
	}
	return size
}
