# M2.1 for AI Coding Tools

> **MiniMax-M2.1** features strong **code understanding**, **multi-turn dialogue**, and **reasoning capabilities**. 

<Note>
  MiniMax Open Platform offers two billing options for text models: Coding Plan and Pay-As-You-Go. You can choose flexibly based on your usage requirements.<br />
  The billing model is determined by the API Key you use. Different types of keys will trigger different billing methods.
</Note>

## Get an API Key

### Coding Plan API Key

If you are subscribed to a coding plan, please use your coding plan API key to access your benefits.

Visit the [**Account/Coding Plan**](https://platform.minimax.io/user-center/payment/coding-plan) page, get your  **Coding Plan API Key** for use in coding tools.
![Api key](https://filecdn.minimax.chat/public/abc85732-74c0-4369-ab96-ef82da0ce2af.png)

### Platform API Key

1. Visit the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key))
2. Click the **"Create new secret key"** button and enter a project name to create a new API key.
3. After creation, the system will display the API key. **Be sure to copy and save it securely**. The key is **shown only once** and cannot be viewed again.

![API-Key](https://filecdn.minimax.chat/public/3ad0f68f-078c-4355-acbe-284a71223f2b.PNG)

## Use MiniMax-M2.1 in Claude Code (Recommended)

### Install Claude Code

Refer to the [Claude Code documentation](https://docs.claude.com/en/docs/claude-code/setup) for installation.

### Configure MiniMax API

<Warning>
  **Important: Clear Anthropic Environment Variables Before Configuration**

  Before configuring, ensure you clear the following Anthropic-related environment variables to avoid conflicts with MiniMax API:

  * `ANTHROPIC_AUTH_TOKEN`
  * `ANTHROPIC_BASE_URL`
</Warning>

1. Edit or create the Claude Code configuration file located at `~/.claude/settings.json`. In this file, add or update the `env` field as shown below.

* The `ANTHROPIC_BASE_URL` should be set based on your location: for international users, use `https://api.minimax.io/anthropic`; for users in China, use `https://api.minimaxi.com/anthropic`.
* Set `<MINIMAX_API_KEY>` to the API key obtained from the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)).

```json  theme={null}
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "<MINIMAX_API_KEY>",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1,
    "ANTHROPIC_MODEL": "MiniMax-M2.1",
    "ANTHROPIC_SMALL_FAST_MODEL": "MiniMax-M2.1",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M2.1",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMax-M2.1",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.1"
  }
}
```

2. After completing the configuration, navigate to your working directory and run the `claude` command in the terminal to start using Claude Code. After startup, select **Trust This Folder** to allow it to access the files in your folder as shown below:

![claude-trust](https://filecdn.minimax.chat/public/ed3b7564-a187-4807-9ae8-218c50182103.PNG)

3. You can now start using Claude Code for development.

### Use M2.1 in Claude Code Extension for VS Code

1. Install Claude Code Extension for VS Code

<img src="https://filecdn.minimax.chat/public/6939e914-b090-4f4f-9c0b-1e394828c23c.jpg" width="80%" />

2. After installation, click **Settings**

![](https://filecdn.minimax.chat/public/d538a295-18e1-4381-ab35-3cfd2fbb7cfc.png)

3. Configure the model to `MiniMax-M2.1`

* In Settings - `Claude Code: Selected Model`, enter `MiniMax-M2.1`

![](https://filecdn.minimax.chat/public/69297d7e-51ee-474b-9ff9-2eb788611972.png)

Or

* Click `Edit in settings.json`, modify `claude-code.selectedModel` to `MiniMax-M2.1` in the configuration file.

<img src="https://filecdn.minimax.chat/public/5c11d0a2-a0b7-4677-ab67-1a3c95d3e35c.png" width="80%" />

4. Configure the environment variables

* If Claude Code is already installed, please refer to the [Claude Code Configuration](/guides/text-ai-coding-tools#configure-minimax-api) to configure the environment variables.
* If Claude Code is not installed, click `Edit in settings.json`

![](https://filecdn.minimax.chat/public/69297d7e-51ee-474b-9ff9-2eb788611972.png)

* Then modify `claudeCode.environmentVariables` to following settings.
* The `ANTHROPIC_BASE_URL` value should be set based on your location: for international users, use `https://api.minimax.io/anthropic`; for users in China, use `https://api.minimaxi.com/anthropic`

```json  theme={null}
"claudeCode.environmentVariables": [
        {
            "name": "ANTHROPIC_BASE_URL",
            "value": "https://api.minimax.io/anthropic"
        },
        {
            "name": "ANTHROPIC_AUTH_TOKEN",
            "value": "<MINIMAX_API_KEY>"
        },
        {
            "name": "API_TIMEOUT_MS",
            "value": "3000000"
        },
        {
            "name": "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "value": "1"
        },
        {
            "name": "ANTHROPIC_MODEL",
            "value": "MiniMax-M2.1"
        },
        {
            "name": "ANTHROPIC_SMALL_FAST_MODEL",
            "value": "MiniMax-M2.1"
        },
        {
            "name": "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "value": "MiniMax-M2.1"
        },
        {
            "name": "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "value": "MiniMax-M2.1"
        },
        {
            "name": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "value": "MiniMax-M2.1"
        }
    ],
```

## Use MiniMax-M2.1 in Cursor

### Install Cursor

1. Download and install Cursor from the [Cursor website](https://cursor.com/agents).
2. Open Cursor, click the **"Settings"** button in the top-right corner to enter the settings page.
3. Click the **"Sign in"** button to log in to your Cursor account.

![cursor-install](https://filecdn.minimax.chat/public/e4755804-3b87-43e3-b595-5c0806f72853.png)

### Configure MiniMax API

<Warning>
  **Important: Clear OpenAI Environment Variables Before Configuration**

  Before configuring, ensure you clear the following OpenAI-related environment variables to avoid conflicts with MiniMax API:

  * `OPENAI_API_KEY`
  * `OPENAI_BASE_URL`
</Warning>

1. Click **"Models"** in the left sidebar to open the model configuration page.
2. Expand the **"API Keys"** section and configure the API information:
   * Enable "Override OpenAI Base URL";
   * Enter the MiniMax base URL (for international users, use `https://api.minimax.io/v1`; for users in China, use `https://api.minimaxi.com/v1`)
3. Paste the API key obtained from the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)) into the OpenAI API Key field.
4. Click the button on the right side of the **“OpenAI API Key”** field.

![cursor-configuration](https://filecdn.minimax.chat/public/09b8444b-1dfd-4aaa-8b3a-7d2918971c6e.png)

5. Click the **"Enable OpenAI API Key"** button in the pop-up window to complete verification.

![cursor-verify](https://filecdn.minimax.chat/public/0b716077-5312-434b-939f-c952e5ed8778.png)

6. In the **Models** section, click the **"View All Models"** button, and then click the **"Add Custom Model"** button.

![](https://filecdn.minimax.chat/public/0c4bebfb-2859-482e-872b-11d6d361bc63.jpeg)

7. Enter the model name **"MiniMax-M2.1"**, then click the **"Add"** button.

![](https://filecdn.minimax.chat/public/39f5ec85-1832-4db6-ac63-630ebe518b34.png)

8. Enable the newly added **"MiniMax-M2.1"** model.
9. Select the **"MiniMax-M2.1"** model in the chat panel and start using "MiniMax-M2.1" for development.

## Use MiniMax-M2.1 in TRAE

### Install and launch TRAE

1. Go to [TRAE's website](https://www.trae.ai/?utm_source=content\&utm_medium=doc_minimax\&utm_campaign=minimax) to install TRAE
2. Click TRAE's icon to launch it
3. The following screen appears at TRAE's first launch
   ![](https://filecdn.minimax.chat/public/ace1b07e-9296-43f1-87ff-e423795716fb.png)
4. Click the Get Started button,TRAE's setup begins

For detailed installation steps, please refer to: [Trae Setup Guide](https://docs.trae.ai/ide/set-up-trae?_lang=en)

### Configure MiniMax API

<Warning>
  **Important: Clear OpenAI Environment Variables Before Configuration**

  Before configuring, ensure you clear the following OpenAI-related environment variables to avoid conflicts with MiniMax API:

  * `OPENAI_API_KEY`
  * `OPENAI_BASE_URL`
</Warning>

In TRAE, you can connenct to MiniMax M2.1 model using API keys obtained from the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key)(For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)).

1. At the top right of the side chat box, click Settings icon > Models
2. The Models window appears
3. Click the + Add Model button
4. The Add Model pop-up appears

<img src="https://filecdn.minimax.chat/public/3dad30c5-c14e-442e-b826-77298befde9a.png" width="60%" />

5. Enter Configuration Details
   * Provider: Select OpenRouter or SiliconFlow
   * Model: Select other models
   * Model ID:  MiniMax M2.1
   * API Key: Input your **[MiniMax API Key](https://platform.minimax.io/user-center/basic-information/interface-key)**

6. Click the **Add Model** button.

## Use MiniMax-M2 in Cline

MiniMax-M2.1 Coming Soon

### Install Cline

1. Open VS Code, click the **Extensions** icon in the left activity bar, and search for "Cline".
2. Click the **"Install"** button to add the extension. After installation, you may need to restart VS Code.

<img src="https://filecdn.minimax.chat/public/88a5d275-752a-4848-8692-3d3e18bae32b.png" width="80%" />

3. Once installed, the Cline icon will appear in the left activity bar.

<img src="https://filecdn.minimax.chat/public/72cea152-c1c0-4c7a-b6ca-d9908a8a9c01.png" width="60%" />

<Warning>
  If you already have Cline installed, please upgrade to version 3.34.1 or higher, and restart both the extension and VS code to ensure proper functionality.
</Warning>

### Configure MiniMax API

<Warning>
  **Important: Clear Anthropic Environment Variables Before Configuration**

  Before configuring, ensure you clear the following Anthropic-related environment variables to avoid conflicts with MiniMax API:

  * `ANTHROPIC_AUTH_TOKEN`
  * `ANTHROPIC_BASE_URL`
</Warning>

1. Click **Use your own API key** to open Cline's configuration page.
2. Under **API Provider**, select **MiniMax**.
3. In **MiniMax Entrypoint**, select the appropriate endpoint based on your location (for international users, select `api.minimax.io`; for users in China, select `api.minimaxi.com`)
4. In **MiniMax API Key**, enter the API key from the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)).
5. In **Model**, select **MiniMax-M2**.
6. Click **Let's go!**, then click **Done** in the top-right corner to save the configuration.

<img src="https://filecdn.minimax.chat/public/1abd9a38-06f6-4745-ae6a-7e1f0d291825.jpeg" width="60%" />

7. Enable the **"Edit"** option box in **"Auto-approve"** section and start using **"MiniMax-M2"** for development.

<img src="https://filecdn.minimax.chat/public/74a0374b-22ed-4cf1-b04c-a51a7f248e39.png" width="80%" />

## Use MiniMax-M2.1 in Kilo Code

### Install Kilo Code

1. Open VS Code, click the **Extensions** icon in the left activity bar, and search for "Kilo Code".

2. Click the **"Install"** button to add the extension. After installation, you may need to restart VS Code.

<img src="https://filecdn.minimax.chat/public/8d69ef21-919d-4751-bc83-a077a14e6f37.png" width="80%" />

### Configure MiniMax API

<Warning>
  **Important: Clear Anthropic Environment Variables Before Configuration**

  Before configuring, ensure you clear the following Anthropic-related environment variables to avoid conflicts with MiniMax API:

  * `ANTHROPIC_AUTH_TOKEN`
  * `ANTHROPIC_BASE_URL`
</Warning>

1. Click Settings to go to the configuration page.

<img src="https://filecdn.minimax.chat/public/827b88f5-8f56-441d-8c81-adb9be4eddc5.png" width="60%" />

2. Under **API Provider**, select **MiniMax**.
3. In **MiniMax Entrypoint**, select the appropriate endpoint based on your location (for international users, select `api.minimax.io`; for users in China, select `api.minimaxi.com`)
4. In **MiniMax API Key**, enter the API key from the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)).
5. In **Model**, select **MiniMax-M2.1**.
6. Click **Save** and then **Done** in the top-right corner to save the configuration.

<img src="https://filecdn.minimax.chat/public/c6432cac-cff9-4751-bf29-efaa1e5f6c8a.png" width="60%" />

7. Start using **MiniMax-M2.1** for development.

<img src="https://filecdn.minimax.chat/public/72e34983-f09c-4a81-a719-5259c674599c.png" width="60%" />

## Use MiniMax-M2.1 in Roo Code

### Install Roo Code

1. Open VS Code, click the **Extensions** icon in the left activity bar, and search for "Roo Code".
2. Click the **"Install"** button to add the extension. After installation, you may need to restart VS Code.

<img src="https://filecdn.minimax.chat/public/9990c533-55e0-4f68-8d7b-e3f494c27b18.png" width="80%" />

### Configure MiniMax API

1. Click Settings to go to the configuration page.

<img src="https://filecdn.minimax.chat/public/1f2271e1-7b95-4feb-a2ac-4d78b6f863e3.png" width="60%" />

2. Under **API Provider**, select **MiniMax**.
3. In **MiniMax Entrypoint**, select the appropriate endpoint based on your location (for international users, select `api.minimax.io`; for users in China, select `api.minimaxi.com`)
4. In **MiniMax API Key**, enter the API key from the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)).
5. In **Model**, select **MiniMax-M2.1**.
6. Click **Save** and then **Done** in the top-right corner to save the configuration.

<img src="https://filecdn.minimax.chat/public/77ecc6c2-ec60-4400-9bb1-1e3fa706627b.png" width="60%" />

7. Start using **MiniMax-M2.1** for development.

<img src="https://filecdn.minimax.chat/public/7ea90b46-ce19-4b5d-ad19-fcb65d053080.png" width="60%" />

## Use MiniMax-M2.1 in Droid

### Install Droid

For Mac/Linux Users：

```bash  theme={null}
curl -fsSL https://app.factory.ai/cli | sh
```

For Windows Users:

```bash  theme={null}
irm https://app.factory.ai/cli/windows | iex
```

For more information, please refer to the [Droid documentation](https://docs.factory.ai/cli/getting-started/quickstart).

### Configure MiniMax API

<Warning>
  **Important: Clear Anthropic Environment Variables Before Configuration**

  Before configuring, ensure you clear the following Anthropic-related environment variables to avoid conflicts with MiniMax API:

  * `ANTHROPIC_AUTH_TOKEN`
  * `ANTHROPIC_BASE_URL`
</Warning>

1. Use following command to edit the configuration file located at `~/.factory/config.json`.

* The `base_url` should be set based on your location: for international users, use `https://api.minimax.io/anthropic`; for users in China, use `https://api.minimaxi.com/anthropic`.
* Set `<MINIMAX_API_KEY>` to the API key obtained from the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)).

<Note>
  Tips:

  * Config `~/.factory/config.json`, NOT `~/.factory/settings.json`
  * Clear the `ANTHROPIC_AUTH_TOKEN` environment variable, otherwise it will override the API key in `.factory/config.json` and cause errors
</Note>

```json  theme={null}
{
    "custom_models": [
        {
            "model_display_name": "MiniMax-M2.1",
            "model": "MiniMax-M2.1",
            "base_url": "https://api.minimax.io/anthropic",
            "api_key": "<MINIMAX_API_KEY>",
            "provider": "anthropic",
            "max_tokens": 64000
        }
    ]
}
```

2. Navigate to your project and start interactive session

```bash  theme={null}
cd /path/to/your/project
droid
```

3. Enter `/model`, select the "**MiniMax-M2.1**" model and use it in Droid.

<img src="https://filecdn.minimax.chat/public/6ff7915e-d9bf-47a5-b008-8d5fdd2d27e7.png" width="60%" />

## Use MiniMax-M2.1 in OpenCode

### Install OpenCode

* Use `curl` to install the OpenCode

```bash  theme={null}
curl -fsSL https://opencode.ai/install | bash
```

* Use `npm` to install the OpenCode

```bash  theme={null}
npm i -g opencode-ai
```

For more information, please refer to the [OpenCode website](https://opencode.ai/)

### Configure MiniMax API

<Warning>
  **Important: Clear Anthropic Environment Variables Before Configuration**

  Before configuring, ensure you clear the following Anthropic-related environment variables to avoid conflicts with MiniMax API:

  * `ANTHROPIC_AUTH_TOKEN`
  * `ANTHROPIC_BASE_URL`
</Warning>

1. Edit or create the OpenCode configuration file located at `~/.config/opencode/opencode.json`. In this file, add or update the env field as shown below.

* The `baseURL` should be set based on your location: for international users, use `https://api.minimax.io/anthropic/v1`; for users in China, use `https://api.minimaxi.com/anthropic/v1`.
* Set `<MINIMAX_API_KEY>` to the API key obtained from the [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)).

```json  theme={null}
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "minimax": {
      "npm": "@ai-sdk/anthropic",
      "options": {
        "baseURL": "https://api.minimax.io/anthropic/v1",
        "apiKey": "<MINIMAX_API_KEY> (Optional)"
      },
      "models": {
        "MiniMax-M2.1": {
          "name": "MiniMax-M2.1"
        }
      }
    }
  }
}
```

**Alternative Authentication Method:**

If you prefer not to add the API key directly to the configuration file, you can authenticate using the interactive login command. Follow these steps:

i. Run the authentication command:

```bash  theme={null}
opencode auth login
```

ii. When prompted, select provider **"Other"**:

<img src="https://filecdn.minimax.chat/public/f616dac4-2ec4-4fc0-aabb-9776d5ec9043.png" width="60%" />

iii. Enter the provider ID as **"minimax"**:

<img src="https://filecdn.minimax.chat/public/4a7d2327-651d-450a-b041-dd17e5ac78ec.png" width="60%" />

iv. Enter your **MiniMax API key** when prompted:

<img src="https://filecdn.minimax.chat/public/74a8214a-8f4a-4019-b335-c9a9717db8da.png" width="80%" />

2. Navigate to your project and start interactive session

```bash  theme={null}
cd /path/to/your/project

opencode
```

3. Enter `/models`, select the "MiniMax-M2.1" model and use it in OpenCode

<img src="https://filecdn.minimax.chat/public/93ece1fb-5841-4056-9980-c872f0f5ebc0.png" width="80%" />

## Use MiniMax-M2.1 in Codex CLI (Not Recommended)

To facilitate integrating models with the Codex CLI, it is recommended to use the `codex-MiniMax-M2` model. While models can interact with files using tools like Bash, this approach is not considered optimal for Agent workflows. For better Agent-based practices, we suggest using platforms such as Claude Code or Cursor.

### Install codex

```bash  theme={null}
npm i -g @openai/codex
```

### Configure MiniMax API

<Warning>
  **Important: Clear OpenAI Environment Variables Before Configuration**

  Before configuring, ensure you clear the following OpenAI-related environment variables to avoid conflicts with MiniMax API:

  * `OPENAI_API_KEY`
  * `OPENAI_BASE_URL`
</Warning>

1. Add the following configuration to the `.codex/config.toml` file.

* The `base_url` should be set based on your location: for international users, use `https://api.minimax.io/v1`; for users in China, use `https://api.minimaxi.com/v1`

```toml  theme={null}
[model_providers.minimax]
name = "MiniMax Chat Completions API"
base_url = "https://api.minimax.io/v1"
env_key = "MINIMAX_API_KEY"
wire_api = "chat"
requires_openai_auth = false
request_max_retries = 4
stream_max_retries = 10
stream_idle_timeout_ms = 300000

[profiles.m2.1]
model = "codex-MiniMax-M2.1"
model_provider = "minimax"
```

2. Set the API key using environment variables in the current terminal session for security reasons. Use the API key obtained from the [**MiniMax Developer Platform**](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)) as the value for `MINIMAX_API_KEY`.

```
export MINIMAX_API_KEY="<MINIMAX_API_KEY>"
```

3. Start the Codex CLI with the specified profile.

```bash  theme={null}
codex --profile m21
```

## Use MiniMax-M2.1 in Grok CLI (Not Recommended)

### Install Grok CLI

1. Use npm to install the Grok CLI globally:

```bash  theme={null}
npm install -g @vibe-kit/grok-cli
```

### Configure MiniMax API

<Warning>
  **Important: Clear OpenAI Environment Variables Before Configuration**

  Before configuring, ensure you clear the following OpenAI-related environment variables to avoid conflicts with MiniMax API:

  * `OPENAI_API_KEY`
  * `OPENAI_BASE_URL`
</Warning>

1. Set the base URL and API key using environment variables.

* The `GROK_BASE_URL` should be set based on your location: for international users, use `https://api.minimax.io/v1`; for users in China, use `https://api.minimaxi.com/v1`.
* Use the API key obtained from the [**MiniMax Developer Platform**](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key)) as the value for `MINIMAX_API_KEY`.

```bash  theme={null}
export GROK_BASE_URL="https://api.minimax.io/v1"  
export GROK_API_KEY="<MINIMAX_API_KEY>"
```

2. Start the Grok CLI with a specified model: MiniMax-M2.1

```bash  theme={null}
grok --model MiniMax-M2.1
```

## Troubleshooting

### API Error

```bash  theme={null}
API Error: Cannot read properties of undefined (reading 'map')
```

Please verify the following:

1. Ensure the API host is set correctly based on your location: for international users, use `https://api.minimax.io`; for users in China, use `https://api.minimaxi.com`
2. Confirm that you have replaced `<MINIMAX_API_KEY>` with your API key from [MiniMax Developer Platform](https://platform.minimax.io/user-center/basic-information/interface-key) (For users in China, visit [MiniMax Developer Platform](https://platform.minimaxi.com/user-center/basic-information/interface-key))
3. If you are using Claude Code or Droid, clear the environment variable: `ANTHROPIC_AUTH_TOKEN`
4. If you are using Claude Code in VS Code or Cursor, please make sure to set the model to `MiniMax-M2.1`. For more details, see: [Using M2.1 in Claude Code for VS Code](/guides/text-ai-coding-tools#use-m2-in-claude-code-for-vs-code)
5. Reached the [rate limit](/guides/rate-limits#3-rate-limits-for-our-api), please wait and try again later.

If the issue persists, feel free to contact us at: [**api@minimax.io**](mailto:api@minimax.io)

## Recommended Reading

<Columns cols={2}>
  <Card title="M2 Tool Use & Interleaved Thinking" icon="book-open" href="/guides/text-m2-function-call" cta="Click here">
    AI models can call external functions to extend their capabilities.
  </Card>

  <Card title="Text Generation" icon="book-open" href="/guides/text-generation" cta="Click here">
    Supports text generation via compatible Anthropic API and OpenAI API.
  </Card>

  <Card title="Compatible Anthropic API (Recommended)" icon="book-open" href="/api-reference/text-anthropic-api" cta="Click here">
    Use Anthropic SDK with MiniMax models
  </Card>

  <Card title="Compatible OpenAI API" icon="book-open" href="/api-reference/text-openai-api" cta="Click here">
    Use OpenAI SDK with MiniMax models
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://platform.minimax.io/docs/llms.txt