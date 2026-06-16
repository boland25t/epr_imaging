; EPR Sampling Tool — Inno Setup installer script
; Build with Inno Setup 6.x:  https://jrsoftware.org/isinfo.php
;
; Before building:
;   1. Run  create_icon.py  to generate app_icon.ico
;   2. Open this file in Inno Setup Compiler and press F9  (or Ctrl+F9 to compile silently)
;   3. Output:  installer\EPRSamplingToolSetup.exe

; ---------------------------------------------------------------------------
; Application metadata
; ---------------------------------------------------------------------------
[Setup]
AppName=EPR Sampling Tool
AppVersion=1.0
AppPublisher=Woods Hole Oceanographic Institution
AppPublisherURL=https://www.whoi.edu
AppSupportURL=https://www.whoi.edu
AppUpdatesURL=https://www.whoi.edu

; Install to the user's local AppData — no admin rights needed
DefaultDirName={localappdata}\EPRSamplingTool
DefaultGroupName=EPR Sampling Tool
DisableProgramGroupPage=yes

; Output
OutputDir=.
OutputBaseFilename=EPRSamplingToolSetup
SetupIconFile=app_icon.ico

; Wizard appearance
WizardStyle=modern
WizardImageFile=..\whoilogolong.png
WizardSmallImageFile=..\whoilogo.png
WizardSizePercent=120

; Compression
Compression=lzma2/max
SolidCompression=yes

; No admin required (installs to %LOCALAPPDATA%)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline

; Uninstaller
UninstallDisplayName=EPR Sampling Tool
UninstallDisplayIcon={app}\sampling_tool.ico
CreateUninstallRegKey=yes

; Minimum OS
MinVersion=10.0

; ---------------------------------------------------------------------------
; Custom welcome / finish text
; ---------------------------------------------------------------------------
[Messages]
WelcomeLabel1=Welcome to the EPR Sampling Tool Setup Wizard
WelcomeLabel2=This wizard will install the EPR Sampling Tool on your computer.%n%nThe installer will automatically download and configure all required software, including Python and all sensor-data processing libraries. An internet connection is required.%n%nClick Next to continue, or Cancel to exit.
FinishedLabel=The EPR Sampling Tool has been installed on your computer.%n%nA shortcut has been placed on your desktop. Click Finish to close this wizard.

; ---------------------------------------------------------------------------
; Files to install
; ---------------------------------------------------------------------------
[Files]

; --- Python source files ---
Source: "..\app.py";                 DestDir: "{app}"; Flags: ignoreversion
Source: "..\chat_panel.py";          DestDir: "{app}"; Flags: ignoreversion
Source: "..\claude_service.py";      DestDir: "{app}"; Flags: ignoreversion
Source: "..\config_service.py";      DestDir: "{app}"; Flags: ignoreversion
Source: "..\dynamicsampling.py";     DestDir: "{app}"; Flags: ignoreversion
Source: "..\main_window.py";         DestDir: "{app}"; Flags: ignoreversion
Source: "..\models.py";              DestDir: "{app}"; Flags: ignoreversion
Source: "..\output_service.py";      DestDir: "{app}"; Flags: ignoreversion
Source: "..\pipeline_service.py";    DestDir: "{app}"; Flags: ignoreversion
Source: "..\point_cloud_pipeline.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\sensor_service.py";      DestDir: "{app}"; Flags: ignoreversion
Source: "..\video_service.py";       DestDir: "{app}"; Flags: ignoreversion
Source: "..\workspace_panel.py";     DestDir: "{app}"; Flags: ignoreversion

; --- Widgets sub-package ---
Source: "..\widgets\*.py";           DestDir: "{app}\widgets"; Flags: ignoreversion

; --- Assets ---
Source: "..\requirements.txt";       DestDir: "{app}"; Flags: ignoreversion
Source: "..\whoilogo.png";           DestDir: "{app}"; Flags: ignoreversion
Source: "..\whoilogolong.png";       DestDir: "{app}"; Flags: ignoreversion
Source: "..\sampling_tool.png";      DestDir: "{app}"; Flags: ignoreversion

; --- Install helper (removed from disk after [Run] completes) ---
Source: "install_helper.ps1";        DestDir: "{app}"; Flags: ignoreversion deleteafterinstall

; ---------------------------------------------------------------------------
; Shortcuts
; (Created after [Run] so launch.bat and sampling_tool.ico already exist)
; ---------------------------------------------------------------------------
[Icons]
; Desktop shortcut
Name: "{autodesktop}\EPR Sampling Tool"; \
      Filename: "{app}\launch.bat"; \
      IconFilename: "{app}\sampling_tool.ico"; \
      WorkingDir: "{app}"; \
      Comment: "EPR Sampling Tool"

; Start Menu entry
Name: "{group}\EPR Sampling Tool"; \
      Filename: "{app}\launch.bat"; \
      IconFilename: "{app}\sampling_tool.ico"; \
      WorkingDir: "{app}"; \
      Comment: "EPR Sampling Tool"

; Uninstaller in Start Menu
Name: "{group}\Uninstall EPR Sampling Tool"; \
      Filename: "{uninstallexe}"

; ---------------------------------------------------------------------------
; Post-install actions
; Run the PowerShell helper that sets up Python, venv, and dependencies.
; This is the step that takes the most time (~3 min on first install).
; ---------------------------------------------------------------------------
[Run]
Filename: "powershell.exe"; \
    Parameters: "-ExecutionPolicy Bypass -NonInteractive -WindowStyle Hidden -File ""{app}\install_helper.ps1"" ""{app}"""; \
    StatusMsg: "Installing Python environment and packages — this takes 2–4 minutes…"; \
    Flags: waituntilterminated

; ---------------------------------------------------------------------------
; Uninstall: remove the entire install directory (venv included)
; ---------------------------------------------------------------------------
[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\widgets\__pycache__"
Type: files;          Name: "{app}\launch.bat"
Type: files;          Name: "{app}\sampling_tool.ico"
Type: files;          Name: "{app}\install_log.txt"

; ---------------------------------------------------------------------------
; Pascal code — adds an "Open install log" button on the Finish page
; if the log file exists (useful for diagnosing pip failures).
; ---------------------------------------------------------------------------
[Code]
var
  ViewLogButton: TNewButton;

procedure ViewLogButtonClick(Sender: TObject);
var
  LogPath: String;
begin
  LogPath := ExpandConstant('{app}\install_log.txt');
  if FileExists(LogPath) then
    ShellExec('', 'notepad.exe', '"' + LogPath + '"', '', SW_SHOW, ewNoWait, 0)
  else
    MsgBox('Install log not found.', mbInformation, MB_OK);
end;

procedure InitializeWizard();
begin
  ViewLogButton           := TNewButton.Create(WizardForm);
  ViewLogButton.Parent    := WizardForm;
  ViewLogButton.Width     := ScaleX(120);
  ViewLogButton.Height    := ScaleY(23);
  ViewLogButton.Left      := WizardForm.CancelButton.Left -
                             ViewLogButton.Width - ScaleX(8);
  ViewLogButton.Top       := WizardForm.CancelButton.Top;
  ViewLogButton.Caption   := 'View Install Log';
  ViewLogButton.Visible   := False;
  ViewLogButton.OnClick   := @ViewLogButtonClick;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  ViewLogButton.Visible := (CurPageID = wpFinished);
end;
