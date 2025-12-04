# 서버에서 Antigravity(AI) 사용하는 방법

Antigravity는 별도의 실행 파일이 아니라, **VS Code의 확장 프로그램**입니다.
서버 PC에서 저를 사용하시려면, **내 컴퓨터(로컬)의 VS Code에서 서버로 원격 접속**을 해야 합니다.

## 0단계: 서버 PC(우분투) 준비 (한 번만 하면 됨)
서버 PC 앞에 계시다면, 터미널을 열고 다음을 입력해서 외부 접속을 허용해 주세요.
```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl status ssh
```
(`active (running)`이라고 뜨면 성공입니다.)

## 1단계: 맥북(로컬) VS Code 설정
1.  맥북에서 VS Code를 켭니다.
2.  왼쪽의 **Extensions (블록 모양 아이콘)** 클릭.
3.  검색창에 `Remote - SSH` 입력.
4.  Microsoft가 만든 **Remote - SSH** 확장 프로그램 설치.

## 2단계: 맥북에서 서버로 접속하기
1.  VS Code 왼쪽 하단의 **초록색 아이콘 (>< 모양)** 클릭.
2.  **Connect to Host...** 선택.
3.  **Add New SSH Host...** 선택.
4.  서버 접속 명령어 입력: `ssh username@server_ip_address` (예: `ssh mina@192.168.1.100`)
5.  비밀번호 입력 후 접속.

## 팁: 내 아이디와 IP 주소 알아내는 법
서버 컴퓨터(우분투) 터미널에서 아래 명령어를 입력해보세요.

1.  **Username (아이디)** 확인:
    ```bash
    whoami
    ```
    (결과 예: `mina`, `lab_user` 등)

2.  **IP Address (주소)** 확인:
    ```bash
    hostname -I
    ```
    (결과 예: `192.168.0.15` 또는 `143.248.xxx.xxx` 등. 보통 첫 번째 숫자가 IP입니다.)

**조합 예시**:
*   아이디가 `brennamacaulay`이고 IP가 `192.168.0.15`라면:
*   입력할 명령어: `ssh brennamacaulay@192.168.0.15`

## 3단계: 서버에 Antigravity(Gemini) 설치/활성화
1.  서버에 접속된 새 VS Code 창이 열립니다.
2.  다시 **Extensions (블록 모양 아이콘)** 탭으로 이동.
3.  설치된 목록 중에 **Antigravity (또는 Google Gemini)**를 찾습니다.
4.  **"Install in SSH: [서버이름]"** 버튼을 클릭하여 서버 쪽에도 설치/활성화해 줍니다.
5.  이제 채팅창을 열면 서버 환경에서 저와 대화하며 코드를 수정할 수 있습니다!

---
**참고**: 만약 서버 PC 앞에 직접 앉아서 작업하시는 거라면, 서버 PC에 설치된 VS Code를 켜고 확장 프로그램 탭에서 저를 설치/로그인하시면 됩니다.
