@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo  Pushing apple-leaf-disease-detection-using-deep-learning
echo ================================================================
echo.

where git >nul 2>&1
if errorlevel 1 (
  echo ERROR: git is not on PATH. Open "Git Bash" or install Git, then retry.
  pause
  exit /b 1
)

echo [1/6] Disabling Git LFS for this repository...
git lfs uninstall >nul 2>&1

echo [2/6] Clearing the staged index (it still holds the 2.6 GB of model files)...
git read-tree --empty
if errorlevel 1 goto :failed

echo [3/6] Re-staging only the files allowed by the new .gitignore...
git add -A
if errorlevel 1 goto :failed

echo.
echo      Files to be committed:
git diff --cached --name-only
echo.

echo [4/6] Creating the commit...
git commit -q -m "Initial commit: Apple leaf disease detection using deep learning" -m "Transfer-learning CNNs (MobileNetV2, VGG16, ResNet50, EfficientNetB0, DenseNet121, InceptionV3) and classical classifiers (SVM, Random Forest, k-NN) over MobileNetV2 features, with an adaptive weighted ensemble and a Streamlit interface." -m "Trained weights and the image dataset are excluded via .gitignore; both are regenerable and together exceed GitHub's size limits."
if errorlevel 1 (
  echo.
  echo      Nothing new to commit - continuing to push.
)

echo [5/6] Pushing to GitHub...
git push -u origin main
if errorlevel 1 goto :pushfailed

echo.
echo ================================================================
echo  SUCCESS
echo  https://github.com/Kirithic01/apple-leaf-disease-detection-using-deep-learning
echo ================================================================
echo.

echo [6/6] Git LFS still holds about 2.6 GB of copied model files in
echo       .git\lfs\objects. Your actual .h5 and .pkl files are NOT
echo       affected - these are duplicates LFS made earlier.
echo.
set /p CLEAN="      Delete that 2.6 GB cache and reclaim the space? [y/N] "
if /i "%CLEAN%"=="y" (
  rmdir /s /q ".git\lfs\objects" 2>nul
  echo       Done - space reclaimed.
) else (
  echo       Skipped. Delete .git\lfs\objects any time to reclaim it.
)

echo.
pause
exit /b 0

:failed
echo.
echo ERROR: a git command failed above. Nothing was pushed.
pause
exit /b 1

:pushfailed
echo.
echo ERROR: the push was rejected. Common causes:
echo   - Not signed in: run  git credential-manager github login
echo   - Wrong account: the repo belongs to Kirithic01
echo.
pause
exit /b 1
