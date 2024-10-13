//Tech Trends Shameer
#define BLYNK_TEMPLATE_ID "TMPL6i70XbIpR"
#define BLYNK_TEMPLATE_NAME "weather app"
#define BLYNK_AUTH_TOKEN "YGaAExpZNK4XitXtFJ72AuqBVOcrc4bb"

#define BLYNK_PRINT Serial
#include <HardwareSerial.h>
#include <AsyncTCP.h>
#include <SPIFFS.h>
#include <ESPAsyncWebServer.h>
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <DHT.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>
#include <time.h>

#define OLED_RESET 4 
Adafruit_SSD1306 display(OLED_RESET);

char auth[] = BLYNK_AUTH_TOKEN;
char ssid[] = "ubnt 24";  // Type your WiFi name
char pass[] = "88118811";  // Type your WiFi password

BlynkTimer timer;
AsyncWebServer server(80);

#define DHTPIN 27 // Connect Out pin to GPIO 27 on ESP32
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define MAX_READINGS 100

struct SensorData {
  float temperature;
  float humidity;
  float uvIndex;
  String timestamp;
  String weather;
};

SensorData readings[MAX_READINGS];
int currentReadingIndex = 0;

// NTP server to get time
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 0; // Set your timezone offset in seconds
const int daylightOffset_sec = 3600; // Set daylight offset if applicable

void printLocalTime() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time");
    return;
  }
  Serial.println(&timeinfo, "%A, %B %d %Y %H:%M:%S");
}

String getTime() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time");
    return "";
  }
  char timeStringBuff[50];
  strftime(timeStringBuff, sizeof(timeStringBuff), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(timeStringBuff);
}

String determineWeather(float temp, float hum, float uv) {
  String weather = "";

  if (temp >= 0 && temp < 20) {
    weather += "Cuaca dingin hujan angin/salju";
  } else if (temp >= 20 && temp < 30) {
    weather += "Cuaca berawan atau hujan ";
  } else if (temp >= 30 && temp < 36) {
    weather += "Cuaca cerah";
  } else {
    weather += "Cuaca panas tinggi ";
  }

  if (hum >= 0 && hum < 30) {
    weather += "kelembaban rendah ";
  } else if (hum >= 30 && hum < 41) {
    weather += "kelembaban normal ";
  } else {
    weather += "kelembaban tinggi ";
  }

  if (uv >= 0 && uv < 2.01) {
    weather += "UV index rendah";
  } else if (uv >= 2.01 && uv < 5.01) {
    weather += "UV index sedang";
  } else if (uv >= 5.01 && uv < 7.01) {
    weather += "UV index lumayan tinggi";
  } else if (uv >= 7.01 && uv < 10.1) {
    weather += "UV index tinggi";
  } else {
    weather += "UV index sangat tinggi";
  }
  return weather;
}

void sendSensor() {
  int sensorValue = analogRead(33);
  float voltage = sensorValue / 1024.0 * 5.0;  // Corrected for ESP32
  float uvIndex = voltage;

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature(); // or dht.readTemperature(true) for Fahrenheit

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // Store the new sensor reading
  readings[currentReadingIndex].temperature = temperature;
  readings[currentReadingIndex].humidity = humidity;
  readings[currentReadingIndex].uvIndex = uvIndex;
  readings[currentReadingIndex].timestamp = getTime();
  readings[currentReadingIndex].weather = determineWeather(temperature, humidity, uvIndex);
  currentReadingIndex = (currentReadingIndex + 1) % MAX_READINGS;

  cuaca(temperature, humidity, uvIndex);

  // You can send any value at any time.
  // Please don't send more than 10 values per second.
  Blynk.virtualWrite(V0, temperature);
  Blynk.virtualWrite(V1, humidity);
  Blynk.virtualWrite(V2, uvIndex);
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.print("    Humidity: ");
  Serial.println(humidity);
  Serial.print("UV: ");
  Serial.println(uvIndex, 2); // Print UV index with two decimal places
}

void cuaca(float temp, float hum, float uv) {
  String weather = determineWeather(temp, hum, uv);

  // Print the determined weather condition
  Serial.print("Kondisi cuaca: ");
  Serial.println(weather);
  hasilserial(weather); // Assuming hasilserial is defined elsewhere
}

void hasilserial(String j1) {
  display.clearDisplay();
  display.setCursor(0, 0); 
  display.setTextSize(1);  // Set a readable font size
  display.setTextColor(SSD1306_WHITE);
  display.println(j1);
  display.display();
  // delay(5000);  // Add a delay of 5 seconds
}

void setup() {   
  Serial.begin(115200);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED){
      delay(500);
      Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  Blynk.begin(auth, ssid, pass);
  dht.begin();
  timer.setInterval(10000L, sendSensor);  // Corrected the interval to 10000ms (10 seconds)
  Wire.begin();
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3C for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for (;;); // Don't proceed, loop forever
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);  // Set the font size
  display.setCursor(0, 0);  // Set the cursor coordinates
  display.println("baru mulai");
  display.display();

  // Setup NTP
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  printLocalTime();

  // Setup AsyncWebServer
  server.on("/sensor", HTTP_GET, [](AsyncWebServerRequest *request){
    String json = "[";
    bool first = true;
    for (int i = 0; i < MAX_READINGS; i++) {
      int index = (currentReadingIndex + i) % MAX_READINGS;
      if (readings[index].temperature == 0 && readings[index].humidity == 0 && readings[index].uvIndex == 0) continue; // Skip uninitialized entries
      if (!first) json += ",";
      first = false;
      json += "{\"temperature\":" + String(readings[index].temperature) + 
              ",\"humidity\":" + String(readings[index].humidity) + 
              ",\"uvIndex\":" + String(readings[index].uvIndex) + 
              ",\"timestamp\":\"" + readings[index].timestamp + "\"" +
              ",\"weather\":\"" + readings[index].weather + "\"}";
    }
    json += "]";
    request->send(200, "application/json", json);
  });

  server.begin();
}

void loop() {
  Blynk.run();
  timer.run();
}
