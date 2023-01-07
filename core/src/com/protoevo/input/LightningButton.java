package com.protoevo.input;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.scenes.scene2d.Touchable;
import com.badlogic.gdx.scenes.scene2d.ui.ImageButton;
import com.badlogic.gdx.scenes.scene2d.utils.Drawable;
import com.badlogic.gdx.scenes.scene2d.utils.TextureRegionDrawable;
import com.badlogic.gdx.utils.Null;
import com.protoevo.ui.InputManager;
import com.protoevo.ui.UI;

public class LightningButton extends ImageButton {

    private static Drawable canStrikeImage, cannotStrikeImage;

    private boolean canStrike = false;

    public LightningButton(InputManager inputManager, float size) {
        super(getDrawable(false));
        setTouchable(Touchable.enabled);

        addListener(event -> {
            if (event.toString().equals("touchDown")) {
                canStrike = !canStrike;
                inputManager.getParticleTracker().setCanTrack(!canStrike);
            }
            return true;
        });

        setHeight(size);
        setWidth(size);
    }

    public boolean canStrike() {
        return canStrike;
    }

    @Override
    protected @Null Drawable getImageDrawable () {
        return getDrawable(canStrike);
    }

    private static Drawable loadDrawable(String path) {
        return new TextureRegionDrawable(new TextureRegion(new Texture(path)));
    }

    private static Drawable getDrawable(boolean state) {
        if (state) {
            if (canStrikeImage == null)
                canStrikeImage = loadDrawable("icons/can_strike.png");
            return canStrikeImage;
        } else {
            if (cannotStrikeImage == null)
                cannotStrikeImage = loadDrawable("icons/cannot_strike.png");
            return cannotStrikeImage;
        }
    }
}
