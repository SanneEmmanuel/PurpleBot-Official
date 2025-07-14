package main

import (
	"encoding/json"
	"log"
	"math"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gorilla/websocket"
)

// --- Static Symbol for stpRNG ---
const staticStpRNG = "stpRNG"

// --- Structs and Shared State ---

type User struct {
	Token         string
	Conn          *websocket.Conn
	OpenPositions int
	LastTrade     string // "BUY", "SELL", or ""
	Mutex         sync.Mutex
}

type Prediction struct {
	PredictedTicks []float64 `json:"predicted_ticks"`
	Confidence     float64   `json:"confidence"`
}

// Application state
var (
	users           = make(map[string]*User)
	usersMutex      sync.Mutex
	currentPrediction Prediction
	predictionMutex sync.Mutex
	confidenceThreshold = 0.88
	libraURL           = os.Getenv("LIBRA_URL") // Example: use ENV for Libra backend
	derivWSURL         = "wss://ws.derivws.com/websockets/v3?app_id=85077"
)

func main() {
	if libraURL == "" {
		libraURL = "http://localhost:5000/predict"
	}

	app := fiber.New()

	// --- ROUTES ---
	app.Post("/login", loginHandler)
	app.Get("/login", loginQueryHandler) // New: login via query param
	app.Get("/status", statusHandler)
	app.Get("/chart-live", chartLiveHandler)

	// --- Background prediction fetcher ---
	go predictionFetcher()

	log.Println("Starting server on :8080")
	log.Fatal(app.Listen(":8080"))
}

// --- /login POST Route (request body) ---

type LoginRequest struct {
	Token string `json:"token"`
}

func loginHandler(c *fiber.Ctx) error {
	var req LoginRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "Invalid request"})
	}
	return loginUser(req.Token, c)
}

// --- /login GET Route (query param) ---

func loginQueryHandler(c *fiber.Ctx) error {
	token := c.Query("api")
	if token == "" {
		return c.Status(http.StatusBadRequest).JSON(fiber.Map{"error": "Missing api token"})
	}
	return loginUser(token, c)
}

// --- Core login logic ---
func loginUser(token string, c *fiber.Ctx) error {
	// Connect to Deriv WebSocket
	ws, _, err := websocket.DefaultDialer.Dial(derivWSURL, nil)
	if err != nil {
		return c.Status(http.StatusInternalServerError).JSON(fiber.Map{"error": "Failed to connect Deriv WebSocket"})
	}

	// Send authorize request
	authReq := map[string]interface{}{"authorize": token}
	if err := ws.WriteJSON(authReq); err != nil {
		ws.Close()
		return c.Status(http.StatusInternalServerError).JSON(fiber.Map{"error": "Failed to send authorize"})
	}

	_, msg, err := ws.ReadMessage()
	if err != nil {
		ws.Close()
		return c.Status(http.StatusInternalServerError).JSON(fiber.Map{"error": "Failed to read authorize response"})
	}
	var authResp map[string]interface{}
	json.Unmarshal(msg, &authResp)
	if authResp["error"] != nil {
		ws.Close()
		return c.Status(http.StatusUnauthorized).JSON(fiber.Map{"error": "Authorization failed"})
	}

	// Store user
	usersMutex.Lock()
	users[token] = &User{
		Token:         token,
		Conn:          ws,
		OpenPositions: 0,
		LastTrade:     "",
	}
	usersMutex.Unlock()
	return c.JSON(fiber.Map{"success": true})
}

// --- /status Route ---

func statusHandler(c *fiber.Ctx) error {
	usersMutex.Lock()
	defer usersMutex.Unlock()
	status := []fiber.Map{}
	for token, user := range users {
		user.Mutex.Lock()
		status = append(status, fiber.Map{
			"token":           token,
			"open_positions":  user.OpenPositions,
			"last_trade":      user.LastTrade,
		})
		user.Mutex.Unlock()
	}
	return c.JSON(status)
}

// --- /chart-live Route ---

func chartLiveHandler(c *fiber.Ctx) error {
	predictionMutex.Lock()
	defer predictionMutex.Unlock()
	return c.JSON(currentPrediction)
}

// --- Prediction Fetcher Goroutine ---

func predictionFetcher() {
	for {
		resp, err := http.Get(libraURL)
		if err != nil {
			log.Println("[Libra] fetch error:", err)
			time.Sleep(3 * time.Second)
			continue
		}
		var pred Prediction
		if err := json.NewDecoder(resp.Body).Decode(&pred); err != nil {
			log.Println("[Libra] decode error:", err)
			resp.Body.Close()
			time.Sleep(3 * time.Second)
			continue
		}
		resp.Body.Close()

		predictionMutex.Lock()
		currentPrediction = pred
		predictionMutex.Unlock()

		executeTrade(pred)
		time.Sleep(3 * time.Second)
	}
}

// --- Decision Engine: executeTrade ---

func executeTrade(pred Prediction) {
	if pred.Confidence < confidenceThreshold {
		return
	}
	direction := detectTrend(pred.PredictedTicks)
	if direction == "HOLD" {
		return
	}
	entryIdx := findBestEntry(pred.PredictedTicks, direction)

	usersMutex.Lock()
	for _, user := range users {
		user.Mutex.Lock()
		if user.LastTrade != direction {
			closeAllPositions(user)
			placeTrade(user, direction, entryIdx)
		} else if user.OpenPositions == 1 {
			placeTrade(user, direction, entryIdx)
		} else if user.OpenPositions == 0 {
			placeTrade(user, direction, entryIdx)
		}
		user.Mutex.Unlock()
	}
	usersMutex.Unlock()
}

// --- Trend Detection ---

func detectTrend(ticks []float64) string {
	if len(ticks) < 2 {
		return "HOLD"
	}
	up, down := true, true
	for i := 1; i < len(ticks); i++ {
		if ticks[i] <= ticks[i-1] {
			up = false
		}
		if ticks[i] >= ticks[i-1] {
			down = false
		}
	}
	switch {
	case up:
		return "BUY"
	case down:
		return "SELL"
	default:
		return "HOLD"
	}
}

// --- Entry Index Finder ---

func findBestEntry(ticks []float64, direction string) int {
	if len(ticks) == 0 {
		return 0
	}
	bestIdx := 0
	switch direction {
	case "BUY":
		minTick := math.MaxFloat64
		for i, tick := range ticks {
			if tick < minTick {
				minTick = tick
				bestIdx = i
			}
		}
	case "SELL":
		maxTick := -math.MaxFloat64
		for i, tick := range ticks {
			if tick > maxTick {
				maxTick = tick
				bestIdx = i
			}
		}
	}
	return bestIdx
}

// --- Trade Placement ---

func placeTrade(user *User, direction string, entryIdx int) {
	// Use static symbol for stpRNG
	proposal := map[string]interface{}{
		"buy": 1,
		"parameters": map[string]interface{}{
			"contract_type":  direction,
			"amount":         1, // Fixed amount for demo
			"currency":       "USD",
			"symbol":         staticStpRNG,
			"duration":       5,
			"duration_unit":  "t",
			"entry_index":    entryIdx,
		},
	}
	if err := user.Conn.WriteJSON(proposal); err != nil {
		log.Println("[Trade] error for user", user.Token, ":", err)
		return
	}
	user.LastTrade = direction
	user.OpenPositions += 1
}

// --- Close All Positions ---

func closeAllPositions(user *User) {
	// Send sell/cancel all (stub)
	sellReq := map[string]interface{}{
		"sell": "all",
	}
	if err := user.Conn.WriteJSON(sellReq); err != nil {
		log.Println("[Close] error for user", user.Token, ":", err)
	}
	user.OpenPositions = 0
	user.LastTrade = ""
}
