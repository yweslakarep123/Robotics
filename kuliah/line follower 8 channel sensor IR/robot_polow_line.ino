//motor a
const int enA = 3; // Pin untuk mengatur kecepatan motor A (PWM)
const int in1 = 2; // Pin untuk mengatur arah motor A (input 1)
const int in2 = 4; // Pin untuk mengatur arah motor A (input 2)


//motor b
const int enB = 5; // Pin untuk mengatur kecepatan motor B (PWM)
const int in3 = 6; // Pin untuk mengatur arah motor B (input 1)
const int in4 = 7; // Pin untuk mengatur arah motor B (input 2)


//sensor
const int s8 = A7; // Pin untuk sensor 8
const int s7 = A6; // Pin untuk sensor 7
const int s6 = A5; // Pin untuk sensor 6
const int s5 = A4; // Pin untuk sensor 5
const int s4 = A3; // Pin untuk sensor 4
const int s3 = A2; // Pin untuk sensor 3
const int s2 = A1; // Pin untuk sensor 2
const int s1 = A0; // Pin untuk sensor 1


// Fungsi agar robot bergerak maju
void gerakroda2() {
  digitalWrite(in1, HIGH); // Mengatur motor A maju
  digitalWrite(in2, LOW);  // Mengatur motor A berhenti
  digitalWrite(in3, HIGH); // Mengatur motor B maju
  digitalWrite(in4, LOW);  // Mengatur motor B berhenti
  analogWrite(enA, 190);    // Mengatur kecepatan motor A (190 dari 255)
  analogWrite(enB, 190);    // Mengatur kecepatan motor B (190 dari 255)
}


// Fungsi agar robot berhenti
void stop() {
  digitalWrite(in1, HIGH); // Mengatur motor A maju
  digitalWrite(in2, LOW);  // Mengatur motor A berhenti
  digitalWrite(in3, HIGH); // Mengatur motor B maju
  digitalWrite(in4, LOW);  // Mengatur motor B berhenti
  analogWrite(enA, 0);     // Mematikan motor A (kecepatan 0)
  analogWrite(enB, 0);     // Mematikan motor B (kecepatan 0)
}


// Fungsi agar robot berbelok kanan
void gerakrodakanan() {
  digitalWrite(in1, HIGH); // Mengatur motor A maju
  digitalWrite(in2, LOW);  // Mengatur motor A berhenti
  digitalWrite(in3, HIGH); // Mengatur motor B maju
  digitalWrite(in4, LOW);  // Mengatur motor B berhenti
  analogWrite(enA, 0);     // Mematikan motor A (kecepatan 0)
  analogWrite(enB, 190);    // Mengatur kecepatan motor B (190 dari 255)
}


// Fungsi agar robot berbelok kanan
void gerakrodakanandikit() {
  digitalWrite(in1, HIGH); // Mengatur motor A maju
  digitalWrite(in2, LOW);  // Mengatur motor A berhenti
  digitalWrite(in3, HIGH); // Mengatur motor B maju
  digitalWrite(in4, LOW);  // Mengatur motor B berhenti
  analogWrite(enA, 0);     // Mematikan motor A (kecepatan 0)
  analogWrite(enB, 160);    // Mengatur kecepatan motor B (160 dari 255)
}


void gerakrodakanandikitbanget() {
  digitalWrite(in1, HIGH); // Mengatur motor A maju
  digitalWrite(in2, LOW);  // Mengatur motor A berhenti
  digitalWrite(in3, HIGH); // Mengatur motor B maju
  digitalWrite(in4, LOW);  // Mengatur motor B berhenti
  analogWrite(enA, 0);     // Mematikan motor A (kecepatan 0)
  analogWrite(enB, 160);    // Mengatur kecepatan motor B (160 dari 255)
}


// Fungsi agar robot berbelok kiri
void gerakrodakiri() {
  digitalWrite(in1, HIGH); // Mengatur motor A maju
  digitalWrite(in2, LOW);  // Mengatur motor A maju
  digitalWrite(in3, HIGH); // Mengatur motor B maju
  digitalWrite(in4, LOW);  // Mengatur motor B maju
  analogWrite(enA, 190);    // Mengatur kecepatan motor A (190 dari 255)
  analogWrite(enB, 0);     // Mematikan motor B (kecepatan 0)
}


void gerakrodakiridikit() {
  digitalWrite(in1, HIGH); // Mengatur motor A maju
  digitalWrite(in2, LOW);  // Mengatur motor A maju
  digitalWrite(in3, HIGH); // Mengatur motor B maju
  digitalWrite(in4, LOW);  // Mengatur motor B maju
  analogWrite(enA, 160);     // Mematikan motor A (kecepatan 160)
  analogWrite(enB, 0);    // Mengatur kecepatan motor B (0 dari 255)
}


void gerakrodakiridikitbanget() {
  digitalWrite(in1, HIGH); // Mengatur motor A maju
  digitalWrite(in2, LOW);  // Mengatur motor A maju
  digitalWrite(in3, HIGH); // Mengatur motor B maju
  digitalWrite(in4, LOW);  // Mengatur motor B maju
  analogWrite(enA, 160);     // Mematikan motor A (kecepatan 160)
  analogWrite(enB, 0);    // Mengatur kecepatan motor B (0 dari 255)
}


void setup() {
  // Inisialisasi pin yang akan digunakan
  Serial.begin(9600);      // Memulai komunikasi serial dengan baud rate 9600


  pinMode(enA, OUTPUT);    // Mengatur pin enA sebagai output
  pinMode(in1, OUTPUT);    // Mengatur pin in1 sebagai output
  pinMode(in2, OUTPUT);    // Mengatur pin in2 sebagai output


  pinMode(enB, OUTPUT);    // Mengatur pin enB sebagai output
  pinMode(in3, OUTPUT);    // Mengatur pin in3 sebagai output
  pinMode(in4, OUTPUT);    // Mengatur pin in4 sebagai output


  pinMode(s8, INPUT);      // Mengatur pin s8 sebagai input
  pinMode(s7, INPUT);      // Mengatur pin s7 sebagai input
  pinMode(s6, INPUT);      // Mengatur pin s6 sebagai input
  pinMode(s5, INPUT);      // Mengatur pin s5 sebagai input
  pinMode(s4, INPUT);      // Mengatur pin s4 sebagai input
  pinMode(s3, INPUT);      // Mengatur pin s3 sebagai input
  pinMode(s2, INPUT);      // Mengatur pin s2 sebagai input
  pinMode(s1, INPUT);      // Mengatur pin s1 sebagai input
}


void loop() {
  // Membaca sensor analog
  int sensor8 = analogRead(s8); // Membaca nilai analog dari sensor 8
  int sensor7 = analogRead(s7); // Membaca nilai analog dari sensor 7
  int sensor6 = analogRead(s6); // Membaca nilai analog dari sensor 6
  int sensor5 = analogRead(s5); // Membaca nilai analog dari sensor 5
  int sensor4 = analogRead(s4); // Membaca nilai analog dari sensor 4
  int sensor3 = analogRead(s3); // Membaca nilai analog dari sensor 3
  int sensor2 = analogRead(s2); // Membaca nilai analog dari sensor 2
  int sensor1 = analogRead(s1); // Membaca nilai analog dari sensor 1


  // Mengambil keputusan berdasarkan nilai sensor untuk arah gerak robot
  if (sensor1 >= 750 && sensor2 >= 750) {
    gerakrodakiri();
  }  
  if (sensor7 >= 750 && sensor8 >= 750) {
    gerakrodakanan();
  }
  if(sensor3 >= 750){
    gerakrodakiridikitbanget();
  }
  if(sensor6 >= 750){
    gerakrodakanandikitbanget();
  }
  if(sensor2 >= 750){
    gerakrodakiridikit();
  }
  if(sensor7 >= 750){
    gerakrodakanandikit();
  }
  if(sensor4 >= 750 && sensor5 >= 750){
    gerakroda2(); // Bergerak maju jika kondisi lain tidak terpenuhi
  }


}
