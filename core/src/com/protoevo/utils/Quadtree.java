package com.protoevo.utils;

import java.util.ArrayList;
import java.util.List;

public class Quadtree {
    private static final int MAX_DEPTH = 5;
    private static final int MAX_OBJECTS = 10;

    private int depth;
    private Rectangle bounds;
    private List<Quadtree> children;
    private List<Rectangle> objects;

    public Quadtree(int depth, Rectangle bounds) {
        this.depth = depth;
        this.bounds = bounds;
        this.children = new ArrayList<>();
        this.objects = new ArrayList<>();
    }

    public Rectangle getBounds() {
        return bounds;
    }

    public boolean isLeaf() {
        return children.size() == 0;
    }

    public List<Quadtree> getChildren() {
        return children;
    }

    public void insert(Rectangle rect) {
        if (children.size() > 0) {
            int index = getIndex(rect);
            if (index != -1) {
                children.get(index).insert(rect);
                return;
            }
        }

        objects.add(rect);

        if (objects.size() > MAX_OBJECTS && depth < MAX_DEPTH) {
            if (children.size() == 0) {
                split();
            }

            int i = 0;
            while (i < objects.size()) {
                int index = getIndex(objects.get(i));
                if (index != -1) {
                    children.get(index).insert(objects.remove(i));
                } else {
                    i++;
                }
            }
        }
    }

    public List<Rectangle> query(Rectangle rect) {
        List<Rectangle> result = new ArrayList<>();
        int index = getIndex(rect);
        if (index != -1 && children.size() > 0) {
            result.addAll(children.get(index).query(rect));
        }
        result.addAll(objects);
        return result;
    }

    private void split() {
        int subWidth = bounds.width / 2;
        int subHeight = bounds.height / 2;
        int x = bounds.x;
        int y = bounds.y;

        children.add(new Quadtree(depth + 1, new Rectangle(x + subWidth, y, subWidth, subHeight)));
        children.add(new Quadtree(depth + 1, new Rectangle(x, y, subWidth, subHeight)));
        children.add(new Quadtree(depth + 1, new Rectangle(x, y + subHeight, subWidth, subHeight)));
        children.add(new Quadtree(depth + 1, new Rectangle(x + subWidth, y + subHeight, subWidth, subHeight)));
    }

    private int getIndex(Rectangle rect) {
        int index = -1;
        double verticalMidpoint = bounds.x + (bounds.width / 2);
        double horizontalMidpoint = bounds.y + (bounds.height / 2);

        boolean topQuadrant = (rect.y < horizontalMidpoint && rect.y + rect.height < horizontalMidpoint);
        boolean bottomQuadrant = (rect.y > horizontalMidpoint);

        if (rect.x < verticalMidpoint && rect.x + rect.width < verticalMidpoint) {
            if (topQuadrant) {
                index = 1;
            } else if (bottomQuadrant) {
                index = 2;
            }
        } else if (rect.x > verticalMidpoint) {
            if (topQuadrant) {
                index = 0;
            } else if (bottomQuadrant) {
                index = 3;
            }
        }

        return index;
    }
}