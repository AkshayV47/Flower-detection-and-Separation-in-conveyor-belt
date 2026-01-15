#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "OPPOReno8T5G";
const char* password = "OPPO8T5G";
const char* remoteIP = "10.226.248.150";  // YOUR PC IP
const int remotePort = 5005;

WiFiUDP udp;

// Same pin config as yours...
// (keep your exact pin config)

void setup() {
  Serial.begin(115200);

  // Camera init (your config)
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = 5; config.pin_d1 = 18; config.pin_d2 = 19; config.pin_d3 = 21;
  config.pin_d4 = 36; config.pin_d5 = 39; config.pin_d6 = 34; config.pin_d7 = 35;
  config.pin_xclk = 0; config.pin_pclk = 22; config.pin_vsync = 25;
  config.pin_href = 23; config.pin_sscb_sda = 26; config.pin_sscb_scl = 27;
  config.pin_pwdn = 32; config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;     // Faster & reliable
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_camera_init(&config);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  Serial.println("WiFi Connected!");
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  // SEND MARKERS + FULL IMAGE AT ONCE (NO FRAGMENTATION)
  udp.beginPacket(remoteIP, remotePort);
  udp.write((uint8_t*)"START", 5);           // Start marker
  udp.write(fb->buf, fb->len);               // Full JPEG
  udp.write((uint8_t*)"END", 3);             // End marker
  udp.endPacket();

  esp_camera_fb_return(fb);
  delay(100);  // ~10 FPS — stable & no corruption
}